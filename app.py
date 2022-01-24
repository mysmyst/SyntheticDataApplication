# IMPORTS   -----------------------------------
from bokeh.models import DataTable, TableColumn, PointDrawTool, ColumnDataSource, CustomJS
from bokeh.plotting import figure, output_file, show, Column
import subprocess
import sys
import ctgan
import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import numpy as np
import random
import os
from faker import Faker
import ruamel.yaml
import json
from smart_open import open
import yaml
from PIL import Image
#Streamlit
import streamlit as st
from streamlit_drawable_canvas import st_canvas
from streamlit_bokeh_events import streamlit_bokeh_events
from streamlit_ace import st_ace
#Sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import metrics
#AIF360
from aif360.sklearn.inprocessing import GridSearchReduction
from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.metrics import disparate_impact_ratio, average_odds_error, generalized_fpr,generalized_fnr, difference
from aif360.algorithms.preprocessing import DisparateImpactRemover
# Gretel
from getpass import getpass
from gretel_client import configure_session, ClientConfig
from gretel_client.helpers import poll
#CTGAN
from ctgan import CTGANSynthesizer,load_demo
#External functions
from functions import *
# from bias import *


#Bias Detections
np.random.seed(0)

from aif360.datasets import GermanDataset
from aif360.metrics import BinaryLabelDatasetMetric
from aif360.algorithms.preprocessing import Reweighing

from IPython.display import Markdown, display


#CTGAN
from ctgan import CTGANSynthesizer
from ctgan import load_demo
from table_evaluator import load_data, TableEvaluator

#TGAN
# import tgan
# from tgan.model import TGANModel

#------------------------------------------------------------------------------------------------
#Global variables
testoutput = []
datatbl=None
visi1,visi2,visi3=False,False,False
#----------------------------------------------------------------------------------------------
#Functions
def dummyfunc():
    pass

#------------------------------------------------------------------------------------------------
#Sidebar
st.sidebar.subheader("Navigation")
page = st.sidebar.radio('What do you want to do?', ('Home', 'Bias Detection and Mitigation',
                                                    'New Data', 'More Data'))
st.sidebar.subheader("Upload Dataset")
dataset = st.sidebar.file_uploader('Upload your dataset here')
if dataset is not None:
    datatbl = pd.read_csv(dataset)
    st.sidebar.write("Uploaded dataset:")
    st.sidebar.dataframe(datatbl.head())
if st.sidebar.button("Graph drawing tool"):
            subprocess.call('python tinker.py')
#----------------------------------------------------------------------------------------------------
#Floating side menu
htmlp2 = '''
            <style>
        .floating-menu {
            font-family: sans-serif;
            background: black;
            padding: 5px;;
            width: 180px;
            z-index: 100;
            position: fixed;
            bottom: 0px;
            right: 0px;
        }

        .floating-menu a,
        .floating-menu h3 {\
            font-size: 0.9em;
            display: block;
            margin: 0 0.5em;
            color: white;
        }
        </style>
        <nav class="floating-menu">
            <h3>GENETHOS🛠️</h3>
            <a href="https://github.com/synthdatagen" target="_blank">Github</a>
            <a href="https://docs.google.com/presentation/d/1H2cDNjSosFWmXeIfXjcs_qUbVFiEZqbJBaqiaUtnAws/edit?usp=sharing" target="_blank">Presentation</a>
            <a href="https://drive.google.com/file/d/1n5dtp2eidtJd4oDUOxxO_YiWuTzWSAq0/view?usp=sharing" target="_blank">Paper</a>

        </nav>
    '''
st.markdown(htmlp2, unsafe_allow_html=True)
#----------------------------------------------------------------------------------------------------
# Navigation of the App
if page == 'Home':
    page_home()

elif page == 'Bias Detection and Mitigation':
    # st.subheader("Coming Soon ⚙")
    # BIAS DETECTION
    st.title("🧬Bias Detection & Mitigation")
    st.markdown("""
    <h5 style="font-style: italic;">This tool gives insights into the bias that the dataset has and provides methods to mitigate it.</h5>""", unsafe_allow_html=True)
    
    st.write("_________")
    st.markdown("<h4>Upload the dataset on the left sidebar and it will show up here</h4>",
                unsafe_allow_html=True)
    if dataset is not None:

        st.write(datatbl)
   
        st.write("_________")
    
        st.markdown("<h4>Bias Detection:</h4>",
                    unsafe_allow_html=True)
        # Step 1: Load Test Train Split
        st.write("Choose bias detection algorithm:")
        biasdetectalgo=st.selectbox("", ["Mean Difference","Disparate Impact","Statistical Parity Difference"])
        
        st.write("Choose bias mitigation algorithm:")
        biasmitialgo=st.selectbox("", ["Reweighing","Disparate Impact Remover","Prejudice Remover"],index=1)
        
        st.write("Choose the column(s) you want to detect bias in (Protected Attribute):")
        colstodetectbias=st.multiselect("", datatbl.columns)
        
        # for i in colstodetectbias:
        #     st.write("enter privileged range for "+f"{i}")
        #     st.text_input(label="lower bound",value=0,key=f"{i}"+"l")
        #     st.text_input(label="upper bound",value=1,key=f"{i}"+"u")
        
        if st.button("Detect & mitigate Bias"):
        
            # Step 1 Load dataset, specifying protected attribute, and split dataset into train and test
            dataset_orig = GermanDataset(
            protected_attribute_names=['age'],           # this dataset also contains protected
                                                        # attribute for "sex" which we do not
                                                        # consider in this evaluation
            privileged_classes=[lambda x: x >= 25],      # age >=25 is considered privileged
            features_to_drop=['personal_status', 'sex'] # ignore sex-related attributes
            )

            #Step 4 Mitigate bias by transforming the original dataset
            dataset_orig_train, dataset_orig_test = dataset_orig.split([0.7], shuffle=True)
            privileged_groups = [{'age': 1}]
            unprivileged_groups = [{'age': 0}]
            
            metric_orig_train = BinaryLabelDatasetMetric(dataset_orig_train, 
                                             unprivileged_groups=unprivileged_groups,
                                             privileged_groups=privileged_groups)
            RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
            dataset_transf_train = RW.fit_transform(dataset_orig_train)

            metric_transf_train = BinaryLabelDatasetMetric(dataset_transf_train, 
                                               unprivileged_groups=unprivileged_groups,
                                               privileged_groups=privileged_groups)

            display(Markdown("#### Original training dataset"))
            st.write("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_orig_train.mean_difference())
            st.write("_________")
            display(Markdown("#### Bias mitigated dataset"))
            st.write("Difference in mean outcomes between unprivileged and privileged groups = %f" % metric_transf_train.mean_difference())
            st.write("_________")

    else:
        st.error("no dataset yet")

elif page == 'New Data':

    st.title("💹New Data")
    st.markdown("""
    <h5 style="font-style: italic;">Generate new data based on your specifications.</h5>""", unsafe_allow_html=True)
    st.write("___________")
    st.subheader("Create your dataset:")

    ncols = int(st.text_input(label="Enter the number of columns in your Dataset", value=1))
    nrows = int(st.text_input(label="Enter number of rows in your Dataset", value=10))
    colnames = [None]*ncols
    coltypes = [None]*ncols
    colsubtypes = [None]*ncols
    colcases = [None]*ncols
    colflags = [[None]*nrows]*ncols
    tblcols = [None]*ncols
    coltypeflag= [None]*ncols
    generationsteps = [1]*ncols
 
    # class tablecol:
        #     def __init__(self, name=None, type=None, subtype=None,cases=[], elements=[None]*nrows, eflags=[None]*nrows):
        #         self.name=name
        #         self.type=type
        #         self.subtype=subtype
        #         self.cases=cases
        #         self.elements=elements
        #         self.eflags=eflags

    st.write("_________")
    st.markdown("<h4>Describe the columns:</h4>",
                unsafe_allow_html=True)

    for i in range(int(ncols)):
        with st.expander(label="Column "+f"{i}"):
            colnames[i] = st.text_input(
                label="Column name", key="coln"+f"{i}")
            coltypes[i] = st.selectbox("Column type:",
                                       ("Numerical", "Categorical(eg: Names, Countries)"), key="colt"+f"{i}")
            if coltypes[i] == "Numerical":
                coltypeflag=0
                colsubtypes[i] = st.selectbox("Describe the column:", (
                    'Number', 'Distribution', 'Python Expression','List','Sequence','Bool'), key="colst"+f"{i}")
            elif coltypes[i] == "Categorical(eg: Names, Countries)":
                coltypeflag=1
                colsubtypes[i] = st.selectbox("Describe the column: "+f"{i}"+" ("+f"{colnames[i]}"+")", (
                    'Blank', 'String', 'Names', 'Countries','Email','URL','Color','Job','ISBN','Credit Card', 'Python Expression','List','Bool'), key="colst"+f"{i}")
            # stpcol1,stpcol2=st.columns(2)
            # with stpcol1:
            #     if st.button("Add step",key=f"{i}"):
            #         generationsteps[i]+=1

            # if generationsteps[i]>1:    
            #     with stpcol2:
            #         if st.button("Remove step"):
            #             generationsteps[i]-=1

            # for j in range(generationsteps[i]):
            #     # if coltypeflag==0:
            #     #     pass
            #     # elif coltypeflag==1:
            #     #     pass
            #     st.write("step"+f"{j+1}",key="genstps"+f"{j}")
            tblcols[i] = colip(colsubtypes[i], nrows, i, tblcols, colnames)

    st.write("____________")
    st.markdown("<h4>Columns added:</h4>",
                unsafe_allow_html=True)
    with st.expander(label=""):
        for i in range(ncols):
            st.write("col "+f"{i}"+" - name: " +
                     colnames[i]+" | type: "+colsubtypes[i]+"\n")
    st.write("____________")
    st.markdown("<h3>Generated Table: </h3>", unsafe_allow_html=True)
    df = pd.DataFrame(tblcols)

    df2 = df.transpose()
    df2.columns = colnames
    rowstoshow = st.slider(label="no. of rows to show in preview",
                           min_value=5, max_value=10)
    st.table(df2.head(rowstoshow))

    if st.button(label="Download Dataset"):
        df2.to_csv("generateddata.csv")
        st.write("Dataset downloaded ✔")
        st.caption("check your project folder")
    

elif page == 'More Data':

    st.title("🗃️More Data")
    st.markdown("""
    <h5 style="font-style: italic;">Generate Synthetic Data using one of the following methods:</h5>""", unsafe_allow_html=True)

    st.write("___________")
    
    st.markdown("<h4>First, upload a dataset from the sidebar to the left</h4>",
                unsafe_allow_html=True)
    if dataset is not None:
        st.write(datatbl)
    else:
        st.error("no dataset yet")
    st.write("_________")

    modelchoice = st.selectbox('Start by selecting a model',
                               ('Click to choose', 'Gretel', 'CTGAN', 'TGAN'))

    # Gretal--------------------------------------------------------------------------------------------------------------

    if modelchoice == 'Gretel':

        # About
        with st.expander("About Gretel"):
            st.image(
                "https://uploads-ssl.webflow.com/5ea8b9202fac2ea6211667a4/5eb59ce904449bf35dded1ab_gretel_wordmark_gradient.svg")

            st.write("""Generate synthetic data to augment your datasets.
                This can help you create AI and ML models that perform
                and generalize better, while reducing algorithmic bias.""")

            st.write("""No need to snapshot production databases to share with your team.
                Define transformations to your data with software,
                and invite team members to subscribe to data feeds in real-time""")

        nrecords = int(st.number_input(label="Number of records to generate", value=1000))


        yaml = ruamel.yaml.YAML()
        yaml.preserve_quotes = True
        with open('input.yaml') as fp:
            data = yaml.load(fp)
        fp.close()
        # no need to iterate further
        data['models'][0]['synthetics']['generate']['num_records'] = nrecords
        with open('input.yaml', 'w') as fp:
            fp = yaml.dump(data, fp)

        if st.button("Generate"):
            st.write("Synthetic Data is being Generated")
            subprocess.call('python gretal.py')
            # creationflags=subprocess.CREATE_NEW_CONSOLE)
            st.write("Synthetic Data is being Generated")
        if st.button("Show Generated Data"):
            if os.path.exists("synthetic_data.csv") == True:
                st.write("Dataset Generated")
                syntheddata = pd.read_csv(
                    "synthetic_data.csv")
                st.dataframe(syntheddata)
            else:
                st.write("Dataset not yet Generated")

        # syntheddata = pd.read_csv(
        #     "D:\Development\Codies\Programming\Python\StreamLit\synthetic_data.csv")
        # cmpr = pd.read_csv(
        #     "D:\Development\Codies\Programming\Python\StreamLit\zraining_data.csv")
        # cmpr1 = cmpr.select_dtypes(include=np.number)
        # cmpr2 = syntheddata.select_dtypes(include=np.number)
        # for i in range(len(cmpr)):
        #     sc1 = cmpr1.iloc[:, i].sample(n=100, replace=True)
        #     sc2 = cmpr2.iloc[:, i].sample(n=100, replace=True)
        # plt.scatter(sc1, sc2, c=['#1f77b4', '#ff7f0e'])
        # plt.pyplot.show()

        if os.path.exists("synthetic_data.csv") == True:
            if st.button("Generate Plots"):
                st.write("Graphs")
                with st.expander("Show"):
                    st.image("1.png")
                    st.image("2.png")
                

    # CTGAN --------------------------------------------------------------------------------------------------------------
    if modelchoice == 'CTGAN':
        with st.container():

            with st.expander("About CTGAN"):
                st.image(
                    "https://sdv.dev/ctgan.svg")

                st.write("""CTGAN is a collection of Deep Learning based Synthetic Data Generators for single table data,
                which are able to learn from real data and generate synthetic clones with high fidelity.
                Currently, this library implements the CTGAN and TVAE models proposed in
                the Modeling Tabular data using Conditional GAN paper.""")
        if st.button('Generate'):
            real_data = load_demo()
            file_path = "./TGAN_credit_risk.csv"
            df = pd.read_csv(file_path)
            df=df.dropna()

        

            # Identifies all the discrete columns

            discrete_columns = [
                'Unnamed: 0',
                'Existing-Account-Status',
                'Credit-History',
                'Purpose',
                'Saving-Acount',
                'Present-Employment',
                'Installment rate',
                'Sex',
                'Guarantors',
                'Residence',
                'Property',
                'Installment',
                'Housing',
                'Existing credits',
                'Job',
                'Num people', 
                'Telephone',
                'foreign worker',
                'Status'
            ]

            # Initiates the CTGANSynthesizer and call its fit method to pass in the table
            
            ctgan = CTGANSynthesizer(epochs=10)
            ctgan.fit(df, discrete_columns)

            #generate synthetic data, 1000 rows of data

            st.write('Synthetic Data Generated')
            synthetic_data = ctgan.sample(1000)
            print(synthetic_data.head(5))

            synthetic_data


    if modelchoice == 'TGAN':
        with st.expander("About TGAN"):

            st.image("https://sdv.dev/rdt.svg")

            st.write("""TGAN is a tabular data synthesizer.
                It can generate fully synthetic data from real data.
                Currently, TGAN can generate numerical columns and categorical columns.""")

        if st.button('Generate'):
            continuous_columns = [2, 5, 13]
            file_path = "./TGAN_credit_risk.csv"
            t_df = pd.read_csv(file_path)
            # tdf.columns
            # tdf=tdf.fillna(0)

            # continuous_columns = [2, 5, 13]
            t_df

            # tgan = TGANModel(
            #     continuous_columns,
            #     output='output',
            #     gpu=None,
            #     max_epoch=5,
            #     steps_per_epoch=10000,
            #     save_checkpoints=True,
            #     restore_session=True,
            #     batch_size=200,
            #     z_dim=200,
            #     noise=0.2,
            #     l2norm=0.00001,
            #     learning_rate=0.001,
            #     num_gen_rnn=100,
            #     num_gen_feature=100,
            #     num_dis_layers=1,
            #     num_dis_hidden=100,
            #     optimizer='AdamOptimizer'
            # )

            # tgan.fit(tdf)
            # num_samples = 1000

            # samples = tgan.sample(num_samples)

            # samples.head(3)
            st.button(label="Save")
          

elif page == 'Test':

    def plot_and_move(df):
        p = figure(x_range=(0, 10), y_range=(0, 10), tools=[],
                   title='Point Draw Tool')

        source = ColumnDataSource(df)

        renderer = p.scatter(x='x', y='y', source=source, size=10)

        draw_tool = PointDrawTool(renderers=[renderer])
        p.add_tools(draw_tool)
        p.toolbar.active_tap = draw_tool

        source.js_on_change("data", CustomJS(
            code="""
                document.dispatchEvent(
                    new CustomEvent("DATA_CHANGED", {detail: cb_obj.data})
                )
            """
        ))

        event_result = streamlit_bokeh_events(
            p, key="foo", events="DATA_CHANGED", refresh_on_update=False, debounce_time=0)

        if event_result:
            df_values = event_result.get("DATA_CHANGED")
            return pd.DataFrame(df_values, index=df_values.pop("index"))
        else:
            return df

    df = pd.DataFrame({
        'x': [1, 5, 9], 'y': [1, 5, 9]
    })

    st.write(plot_and_move(df))

#-----------------------------------------------------------------------------------------------------
#Misc code to hide unnecessary elements
hide_streamlit_style = """
            <style>
            # MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """

st.markdown(hide_streamlit_style, unsafe_allow_html=True)

hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)
# -------------------------------------------------------------------------------------
#Interactive drawing tool test code
# output_file("tools_point_draw.html")
# p = figure(x_range=(0, 10), y_range=(0, 10), tools=[],
#            title='Point Draw Tool')
# p.background_fill_color = 'lightgrey'
# source = ColumnDataSource({
#     'x': [1, 5, 9], 'y': [1, 5, 9], 'color': ['red', 'green', 'yellow']
# })
# renderer = p.scatter(x='x', y='y', source=source, color='color', size=10)
# columns = [TableColumn(field="x", title="x"),
#            TableColumn(field="y", title="y"),
#            TableColumn(field='color', title='color')]
# table = DataTable(source=source, columns=columns, editable=True, height=200)

# draw_tool = PointDrawTool(renderers=[renderer], empty_value='black')
# p.add_tools(draw_tool)
# p.toolbar.active_tap = draw_tool
# print(str())
# show(Column(p, table))
#----------------------------------------------------------------------------------------