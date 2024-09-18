from taipy.gui import Gui, notify, invoke_long_callback, Html
from pydantic import BaseModel, Field, create_model
from typing import Type, Any, get_type_hints
import dspy
from dspy.predict import KNN
from dspy.teleprompt import KNNFewShot
from dspy.datasets.dataset import Dataset
from dotenv import load_dotenv
import nest_asyncio
import os
import json
import pandas as pd
from langchain_openai import AzureChatOpenAI
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams
import dash
from dash import Dash, dash_table, html, dcc
from dash.dependencies import Input, Output
import threading
from flask import Flask
import requests
import tiktoken


load_dotenv()

nest_asyncio.apply()

server = Flask(__name__)


class Examples(Dataset):
    def __init__(self, file_path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        self._train = pd.read_csv(file_path, sep=";").to_dict(orient='records')


class Extraction(BaseModel):  
    proposal_by_the_applicant: str = Field(description="The pharma company suggests their own idea for a given topic")
    chmp_agreement: str = Field(description="CHMP can agree, partially agree or disagree with the proposal of the applicant, give some short context")
    chmp_comments_suggestions_caveats: str = Field(description="CHMP additional comments related to the proposal, not previously mentioned")
    #reference_in_text: int = Field(description="The question which contains this information like for example: Extracted from Question 1")  
  

class ExtractionSet(BaseModel):
    comparator: Extraction = Field(description="Can only be another drug or placebo or dietary intervention")  
    primary_endpoints: Extraction = Field(description="Can only be a target or goal which can be reached at the end of the trial")  
    secondary_endpoints: Extraction = Field(description="Can only be a target or goal which can be reached at the end of the trial and which is of lower importance")
    #trial_design_description: Extraction = Field(description="Can only be a scientificly structured hypothesis or process plan")
    #safety_issues_and_plans: Extraction = Field(description="Can only be a description of model evaluations, mitigation strategies, safety concerns")
    #dosing_regimen: Extraction = Field(description="Can only be a proposed dosing or exposure regimen")
    #trial_population_diagnosis_or_condition: Extraction = Field(description="Can only be a clinical indication or disease")
    #primary_statistical_analysis: Extraction = Field(description="Can only be mathematical terms")
    #interim_analysis: Extraction = Field(description="interim analysis description where an interim analysis is analysis of data before data collection has been completed")
    #non_inferiority_equivalence_margin: Extraction = Field(description="include the size of the margin, the statistical or clinical justification")


class ModelManager:
    def __init__(self):
        self.models = {
            "ExtractionSet": ExtractionSet,
            "Extraction": Extraction
        }


    def add_field(self, model_name: str, field_name: str, field_type: Type[Any], description: str):
        model_class = self.models[model_name]
        type_hints = get_type_hints(model_class)
        fields = model_class.model_fields.copy()

        fields[field_name] = Field(description=description)
        type_hints[field_name] = field_type
        
        updated_model = create_model(
            model_class.__name__,
            __base__=model_class.__base__,
            **{name: (type_hints[name], field) for name, field in fields.items()}
        )
        
        self.models[model_name] = updated_model
        
        if model_name == "Extraction":
            self._update_extraction_set()

        return updated_model


    def delete_field(self, model_name: str, field_name: str):
        model_class = self.models[model_name]
        type_hints = get_type_hints(model_class)
        fields = model_class.model_fields.copy()
        
        if field_name in fields:
            del fields[field_name]
        if field_name in type_hints:
            del type_hints[field_name]
        
        updated_model = create_model(
            model_class.__name__,
            __base__=model_class.__base__,
            **{name: (type_hints.get(name, Any), field) for name, field in fields.items()}
        )
        
        self.models[model_name] = updated_model
        
        if model_name == "Extraction":
            self._update_extraction_set()
        
        return updated_model


    def _update_extraction_set(self):
        extraction_model = self.models["Extraction"]
        extraction_set_model = self.models["ExtractionSet"]
        
        updated_fields = {}
        for name, field in extraction_set_model.model_fields.items():
            updated_fields[name] = (extraction_model, field)

        
        updated_extraction_set = create_model(
            "ExtractionSet",
            __base__=BaseModel,
            **updated_fields
        )
        self.models["ExtractionSet"] = updated_extraction_set


    def get_model_fields(self, model_name: str):
        return list(self.models[model_name].model_fields.keys())


def create_structure(extraction_model):
    class Structure(dspy.Signature):
        """You are an AI assistant that helps EMA officers to extract short information from Scientific Advice Letters. You only briefly and concisely answer based on the information found in the letter, not based on your general knowledge. If you dont find the answer, give back 'n/a'. In the Final Advice Letter that follows, a company asks EMA for Scientific Advice on the development of a new product. The letter includes a conversation between the company and the CHMP with the details on the clinical studies to be performed for the development of the medicinal product."""
        input: str = dspy.InputField()
        output: extraction_model = dspy.OutputField()
    return Structure


def create_mechanism(structure):
    class Mechanism(dspy.Module):
        def __init__(self):
            super().__init__()
            self.extract = dspy.TypedPredictor(structure)

        def forward(self, input):
            return self.extract(input=input).output
    return Mechanism()


def update_model_fields(state):
    state.model_fields = "\n".join(f"- {field}" for field in model_manager.get_model_fields("ExtractionSet"))
    state.model_fields2 = "\n".join(f"- {field}" for field in model_manager.get_model_fields("Extraction"))


def field_addition(state):
    global Structure, ZeroShot, teleprompter, KNN_FewShot
    if state.field_name and state.field_description:
        model_name = "ExtractionSet" if state.level_choice == "Level 1" else "Extraction"
        field_type = model_manager.models["Extraction"] if state.level_choice == "Level 1" else str
        model_manager.add_field(model_name, state.field_name, field_type, state.field_description)
        
        Structure = create_structure(model_manager.models["ExtractionSet"])
        ZeroShot = create_mechanism(Structure)
        KNN_FewShot = teleprompter.compile(student=create_mechanism(Structure), trainset=examples.train)
        
        update_model_fields(state)
        notify(state, "success", f"Field '{state.field_name}' added successfully!")
        state.field_name = ""
        state.field_description = ""
    else:
        notify(state, "error", "Please provide both field name and description.")


def field_deletion(state):
    global Structure, ZeroShot, teleprompter, KNN_FewShot
    if state.field_name:

        model_name = "ExtractionSet" if state.level_choice == "Level 1" else "Extraction"
        model_manager.delete_field(model_name, state.field_name)
        
        Structure = create_structure(model_manager.models["ExtractionSet"])
        ZeroShot = create_mechanism(Structure)
        KNN_FewShot = teleprompter.compile(student=create_mechanism(Structure), trainset=examples.train)
    

        update_model_fields(state)
        notify(state, "success", f"Field '{state.field_name}' deleted successfully!")
        state.field_name = ""
        state.field_description = ""
    else:
        notify(state, "error", "Please provide a field name to delete.")
    

class Azure_OpenAI(DeepEvalBaseLLM):
    def __init__(
        self,
        model
    ):
        self.model = model

    def load_model(self):
        return self.model


    def generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content


    async def a_generate(self, prompt: str) -> str:
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content


    def get_model_name(self):
        return "Custom Azure OpenAI Model"
    

def nested_json_to_dataframe(nested_dict):
    level1 = []
    level2 = []
    extraction = []

    for key1, value1 in nested_dict.items():
        if isinstance(value1, dict):
            for key2, value2 in value1.items():
                level1.append(key1)
                level2.append(key2)
                extraction.append(value2)
        else:
            level1.append(key1)
            level2.append('')
            extraction.append(value1)

    df = pd.DataFrame({
        'level1': level1,
        'level2': level2,
        'extraction': extraction
    })

    return df


def create_dash_app(server):
    global dash_app
    dash_app = dash.Dash(__name__, server=server, url_base_pathname='/dash/')
    dash_app.layout = html.Div([
        dcc.Interval(id='interval-component', interval=1000, n_intervals=0),
        html.Div(id='dash-table-container')
    ])

    @dash_app.callback(
        Output('dash-table-container', 'children'),
        Input('interval-component', 'n_intervals')
    )
    def update_dash_table(_):
        if dash_update_event.is_set():
            dash_update_event.clear()
            if hasattr(dash_app, 'final_result'):
                column_defs = [
                    {"name": "Level 1", "id": "level1"},
                    {"name": "Level 2", "id": "level2"},
                    {"name": "Extraction", "id": "extraction"},
                ]
                
                has_score = 'score' in dash_app.final_result.columns
                if has_score:
                    column_defs.append({"name": "Score", "id": "score"})

                # Define table properties
                table_props = {
                    'columns': column_defs,
                    'data': dash_app.final_result.to_dict('records'),
                    'style_table': {
                        'width': '100%',  # Ensure table width is responsive
                        'overflowX': 'auto',  # Allow horizontal scrolling if needed
                        'overflowY': 'auto',  # Allow vertical scrolling if needed
                    },
                    'style_cell': {
                        'overflow': 'hidden',
                        'textOverflow': 'ellipsis',
                        'padding': '10px',
                        'fontSize': 15, 
                        'fontFamily': 'sans-serif',        
                        'textAlign': 'left',
                        'minWidth': '0px',  # Allow flexible column sizing
                        'maxWidth': '200px',  # Optional max width for general columns
                    },
                    'style_data': {
                        'whiteSpace': 'normal',
                        'height': 'auto',
                    },
                    'cell_selectable': False,
                    'row_selectable': False,
                    'fixed_rows': {'headers': True},  # Keep headers fixed
                }

                # Handle score column-specific properties
                if has_score:
                    # Tooltips for score column
                    table_props['tooltip_data'] = [
                        {
                            'score': {'value': str(row.get('reason', '')), 'type': 'markdown'}
                        } for row in dash_app.final_result.to_dict('records')
                    ]
                    table_props['tooltip_duration'] = None

                    # Adjust column widths: narrower for level1, level2, and score; wider for extraction
                    table_props['style_cell_conditional'] = [
                        {
                            'if': {'column_id': 'level1'},
                            'minWidth': '100px',  # Narrow for Level 1
                            'width': '100px',
                            'maxWidth': '100px',
                        },
                        {
                            'if': {'column_id': 'level2'},
                            'minWidth': '100px',  # Narrow for Level 2
                            'width': '100px',
                            'maxWidth': '100px',
                        },
                        {
                            'if': {'column_id': 'extraction'},
                            'minWidth': '150px',  # Make the Extraction column wider
                            'width': '150px',
                            'maxWidth': '150px',  # No max width for extraction
                        },
                        {
                            'if': {'column_id': 'score'},
                            'minWidth': '80px',  # Slightly narrow for Score
                            'width': '80px',
                            'maxWidth': '80px',
                            'textAlign': 'center',
                        }
                    ]

                    # Conditional formatting for score values
                    table_props['style_data_conditional'] = [
                        {
                            'if': {
                                'column_id': 'score',
                                'filter_query': '{score} < 0.4'
                            },
                            'backgroundColor': 'rgb(255, 178, 178)',
                            'color': 'black',
                            'fontWeight': 'bold',
                            'borderRadius': '12px',
                            'padding': '6px 10px',
                            'margin': '4px',
                        },
                        {
                            'if': {
                                'column_id': 'score',
                                'filter_query': '{score} >= 0.4 && {score} < 0.7'
                            },
                            'backgroundColor': 'rgb(255, 245, 157)',
                            'color': 'black',
                            'fontWeight': 'bold',
                            'borderRadius': '12px',
                            'padding': '6px 10px',
                            'margin': '4px',
                        },
                        {
                            'if': {
                                'column_id': 'score',
                                'filter_query': '{score} >= 0.7'
                            },
                            'backgroundColor': 'rgb(173, 235, 173)',
                            'color': 'black',
                            'fontWeight': 'bold',
                            'borderRadius': '12px',
                            'padding': '6px 10px',
                            'margin': '4px',
                        },
                    ]

                    # Custom CSS to avoid misalignment during hover or tooltip display
                    table_props['css'] = [{
                        'selector': '.dash-table-tooltip',
                        'rule': 'background-color: white; color: black;',
                    }, {
                        'selector': 'tr:hover',
                        'rule': 'background-color: inherit !important;',  # Prevent hover misalignment
                    }]

                return dash_table.DataTable(**table_props)

        return dash.no_update


def calculate_tokens(document_input):
    encoding = tiktoken.get_encoding("cl100k_base")
    tokens = encoding.encode(document_input)
    token_count_input = len(tokens)

    return token_count_input


def run_flask():
    server.run(port=8050)


@server.route('/trigger-dash-update', methods=['POST'])
def trigger_dash_update():
    global dash_update_event
    dash_update_event.set()
    return 'OK', 200


def trigger_dash_update(state):
    requests.post('http://localhost:8050/trigger-dash-update')


def zero_shot(document_input):
    response_model = ZeroShot(input=document_input)        
    return response_model


def validate(document_input, result, validation_choice):
    document = document_input
    df = result

    if validation_choice == "Scoring ON":
        scores = []
        reasons = []

        azure_openai = Azure_OpenAI(model=custom_model)
        
        metric = GEval(
            model=azure_openai,
            name="Extraction Validation",
            evaluation_steps=[
                "Determine coherence (0-1), whether the 'actual output' fits logically and factually to the input question.",
                "Determine faithfulness (0-1), whether the 'actual output' can be found in the text.",
                "Determine consistency (0-1), if the 'actual output' suggests that the info is not contained in the document (NA), that it is indeed not contained in the document.",
                "Do not penalize brief outputs or ommittance of supplementary information in the actual output.",
                "Do not penalize NA if correct (indeed not found in context)"
            ],
            evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.CONTEXT, LLMTestCaseParams.ACTUAL_OUTPUT]
        )

        for index, row in df.iterrows():
            level1 = row['level1']
            level2 = row['level2']
            extraction = row['extraction']
            
            query = f"What is the {level2} for {level1} in the input document?"
            test_case = LLMTestCase(
                input=query,
                actual_output=extraction,
                context=[document]
            )
            
            metric.measure(test_case)
            
            scores.append(metric.score)
            reasons.append(metric.reason)

        result_df = pd.DataFrame({
            'score': scores,
            'reason': reasons
        })

        validation_result = pd.concat([df, result_df], axis=1)
        validation_result['score'] = validation_result['score'].apply(lambda x: f"{x:.2f}")
    else:
        validation_result = df

    return validation_result


def zero_shot_and_validate(document_input, validation_choice):
    response_model = zero_shot(document_input)
    result_dict = json.loads(json.dumps(response_model.dict(), indent=2))
    extraction_output = nested_json_to_dataframe(result_dict)
    validation_result = validate(document_input, extraction_output, validation_choice)
    
    return validation_result, response_model

        
def knn_optimization_and_validate(document_input, validation_choice): 
    response_model = KNN_FewShot(input=document_input)
    result_dict = json.loads(json.dumps(response_model.dict(), indent=2))
    extraction_output = nested_json_to_dataframe(result_dict)
    validation_result = validate(document_input, extraction_output, validation_choice)
    
    return validation_result, response_model


def on_extract_action(state):
    if state.document_input:
        state.token_count_input = calculate_tokens(state.document_input)
        if state.token_count_input >= 6144:
            notify(state, "error", "Input document exceeds token limit")
        else:
            notify(state, "info", "Extraction and validation started")
            if state.knn_choice == "KNN FewShot OFF": 
                invoke_long_callback(state, zero_shot_and_validate, [state.document_input, state.validation_choice], heavy_function_status)
            else:
                invoke_long_callback(state, knn_optimization_and_validate, [state.document_input, state.validation_choice], heavy_function_status)
            trigger_dash_update(state)
    else:
        notify(state, "error", "Please provide a document to extract from.")


def heavy_function_status(state, status, results):
    if status and results:
        validation_result, response_model = results
        notify(state, "success", "Extraction and validation completed successfully!")
        state.final_result = validation_result
        
        zero_shot_text = json.dumps(response_model.dict(), indent=2)
        prompt = get_latest_content()
        state.token_count_output = calculate_tokens(zero_shot_text)
        state.token_count_input = calculate_tokens(prompt)
        
        global dash_app, dash_update_event
        if dash_app:
            dash_app.final_result = state.final_result
            dash_app.has_score = 'score' in state.final_result.columns
            dash_update_event.set()
    else:
        notify(state, "error", "The extraction or validation process has failed")


def write_to_file(content, filename="prompt.txt"):
    with open(filename, "w", encoding="utf-8") as file:
        file.write(content)
    return filename


def on_download(state):
    content = get_latest_content()
    filename = write_to_file(content)
    state.download_path = os.path.abspath(filename)
    notify(state, "success", f"File '{filename}' is ready for download!")


def get_latest_content():
    return lm.history[-1]['prompt'] if lm.history else ""


def get_doc_names():
    fals = pd.read_pickle("fals_content.pkl")
    return fals.iloc[:, 0].tolist()


def get_doc_content(doc_name):
    fals = pd.read_pickle("fals_content.pkl")
    content = fals[fals.iloc[:, 0] == doc_name].iloc[0, 1]
    return content


def read_file(file_name):
    try:
        with open(file_name, 'r', encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"


def on_change(state, var_name, var_value):
    if var_name == "document_chosen" and var_value is not None:
        state.document_input = get_doc_content(var_value)


# Global variables
lm = dspy.AzureOpenAI(
    api_base=os.getenv("AZURE_OPENAI_API_BASE"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    model="gpt-4",
    max_tokens=2048,
    temperature=0)
custom_model = AzureChatOpenAI(
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_deployment="gpt-4",
    azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    temperature=0
)
dspy.settings.configure(lm=lm)
model_manager = ModelManager()
Structure = create_structure(model_manager.models["ExtractionSet"])
ZeroShot = create_mechanism(Structure)
examples = Examples(file_path="KNN_examples.csv", input_keys=["input"])
teleprompter = KNNFewShot(k=3, trainset=examples.train, max_bootstrapped_demos=0)
KNN_FewShot = teleprompter.compile(student=ZeroShot, trainset=examples.train)
field_name = ""
field_description = ""
model_fields = "\n".join(f"- {field}" for field in list(ExtractionSet.__annotations__.keys())) 
model_fields2 = "\n".join(f"- {field}" for field in list(Extraction.__annotations__.keys())) 
extraction_output = pd.DataFrame(columns=["level1", "level2", "extraction"])
final_result = pd.DataFrame(columns=["level1", "level2", "extraction"])
dash_app = None
dash_update_event = threading.Event()
level_choice = "Level 1"
validation_choice = "Scoring ON"
knn_choice = "KNN FewShot OFF"
download_path = None
token_count_input = 0
token_count_output = 0
doc_list = get_doc_names()
document_chosen = None
document_input = "No document selected"
create_dash_app(server)
flask_thread = threading.Thread(target=run_flask)
flask_thread.start()

page = Html("""
<taipy:layout columns="1 1 2" gap="40px" style="height: 100vh;">
  <taipy:part style="height: 100%; overflow-y: auto;">
    <div style="height: 100%;">
      <h4>Change fields</h4>
      <p>Choose level and enter field name and description for adding to or deleting it from the extraction.</p>
        <taipy:toggle lov="Level 1;Level 2" allow_unselect="True">{level_choice}</taipy:toggle>
      <p>
        Field Name: <taipy:input> {field_name}</taipy:input>
      </p>
      <p>
        Field Description: <taipy:input> {field_description}</taipy:input>
      </p>
      <taipy:button on_action="field_addition">Add Field</taipy:button>
      <taipy:button on_action="field_deletion">Delete Field</taipy:button>
      <h4>Current Model Fields</h4>
      <taipy:expandable title="Level 1">
        <taipy:text mode="markdown">{model_fields}</taipy:text>
      </taipy:expandable>
      <taipy:expandable title="Level 2">
        <taipy:text mode="markdown">{model_fields2}</taipy:text>
      </taipy:expandable>
    </div>
  </taipy:part>
  <taipy:part style="height: 100%; overflow-y: auto;">
    <div style="height: 100%;">
      <h4>Document Extraction</h4>
      <p>Select / paste your document here:</p>
      <taipy:selector lov={doc_list} dropdown>{document_chosen}</taipy:selector>
      <taipy:input>{document_input}</taipy:input>
            <br></br>
            <br></br>
      <taipy:toggle lov="KNN FewShot ON;KNN FewShot OFF" allow_unselect="True">{knn_choice}</taipy:toggle>
      <br></br>
            <br></br>
      <taipy:toggle lov="Scoring ON;Scoring OFF" allow_unselect="True">{validation_choice}</taipy:toggle>
      <br></br>
            <br></br>
      <taipy:button on_action="on_extract_action">Extract</taipy:button>
      <br></br>
            <div id="root">
            <div class="MuiBox-root css-0" style="display: flex;">
            <main class="MuiBox-root css-7g81c7">
            <div class="jsx-parser">
            <div class="taipy-part card card-bg MuiBox-root css-0">
            <div class="md-para p1 mb1">
                  <b>Model token usage and limits</b>
                  <br></br>
                  <br></br>
                  <taipy:text>{token_count_input} / 6144 tokens in input</taipy:text>
                  <br></br><em>(dependent on document size, knn few shot and fields in level1 and level2)</em>
                  <br></br>
                  <br></br>
                  <taipy:text>{token_count_output} / 2048 tokens in output</taipy:text>
                  <br></br><em>(dependent on fields in level1 and level2)</em>
            </div></div></div></main></div></div>
    </div>
  </taipy:part>
  <taipy:part style="height: 100%; overflow-y: auto;">
    <div style="height: 100%;">
      <h4>Extraction Output</h4>
      <taipy:button on_action="on_download">Save</taipy:button>
      <taipy:file_download label="Download prompt for this extraction" name="prompt.txt">{download_path}</taipy:file_download>
      <br></br>
      <br></br>
      <br></br>
      <iframe src="http://localhost:8050/dash/" style="border: none; width: 850px; height: 850px; overflow-y: auto; overflow-x: hidden;"></iframe>
    </div>
  </taipy:part>
</taipy:layout>
""")

gui = Gui(page=page)
if __name__ == "__main__":
    print(f"Dash app running on port 8050")
    print(f"Taipy app running on port 5000")
    print(f"Please navigate to http://localhost:5000 in your web browser")
    gui.run(use_reloader=False, port=5000, light_theme=True, dark_theme=False, dark_mode=False, title="Prompt programming")