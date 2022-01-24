import streamlit as st
from faker import Faker
from streamlit_ace import st_ace
import random
import subprocess
from scipy.stats import norm,expon
#from tink3 import *

def colip(coltype, nrows, i, tblcols,colnames):
    # for cc in colnames:
    #     pass
    #     # eval()


    j=0
    fake = Faker()
    element=None
    if coltype == "Number":
        x = int(st.text_input(label="enter an integer", value=0, key="int"+str(i)+str(j)))
        return [x]*nrows

    if coltype == "String":
        x2 = st.text_input(label="enter a string",
                           value="foo", key="str"+str(i)+str(j))
        return [x2]*nrows

    if coltype == "Sequence":
        start = int(st.text_input(
            label="enter start value", value=0, key="sta"+str(i)+str(j)))
        increment = int(st.text_input(
            label="enter increment value", value=1, key="inc"+str(i)+str(j)))
        op = [None]*nrows
        for i in range(nrows):
            op[i] = start+(i*increment)
        return op

    if coltype == "Names":
        nametype = st.selectbox(
            "select name type", ('Full name', 'First name', 'First name - male', 'First name - female', 'Last name'), key="nt"+str(i)+str(j))
        op = [None]*nrows
        if nametype == 'Full name':
            for i in range(nrows):
                op[i] = fake.name()
        elif nametype == 'First name':
            for i in range(nrows):
                op[i] = fake.first_name()
        elif nametype == 'First name - male':
            for i in range(nrows):
                op[i] = fake.first_name_male()
        elif nametype == 'First name - female':
            for i in range(nrows):
                op[i] = fake.first_name_female()
        elif nametype == 'Last name':
            for i in range(nrows):
                op[i] = fake.last_name()
        return op

    if coltype == "Countries":
        op = [None]*nrows
        for i in range(nrows):
            op[i] = fake.country()
        return op

    
    if coltype == "URL":
        op = [None]*nrows
        for i in range(nrows):
            op[i] = fake.url()
        return op

    if coltype == "Bool":
        op = [None]*nrows
        ptrue=int(st.text_input(label="Probaility of getting True", value=50, key="ptrue"+str(i)+str(j)))
        for i in range(nrows):
            op[i] = fake.boolean(chance_of_getting_true=ptrue)
        return op

    if coltype == "Job":
        op = [None]*nrows
        for i in range(nrows):
            op[i] = fake.job()
        return op

    if coltype == "ISBN":
        op = [None]*nrows
        
        isbntype=st.selectbox(label="ISBN type",options=("10","13"),key="cctype"+str(i)+str(j))
        
        if isbntype=="10":
            for i in range(nrows):
                op[i] = fake.isbn10()
        if isbntype=="13":
            for i in range(nrows):
                op[i] = fake.isbn13()
        return op

    if coltype == "Color":
        op = [None]*nrows
        for i in range(nrows):
            op[i] = fake.color_name()
        return op
    
    if coltype == "Email":
        op = [None]*nrows
        for i in range(nrows):
            op[i] = fake.email()
        return op

    if coltype == "Credit Card":
        cctype=st.selectbox(label="value type",options=("Expiry","CVC","Full"), key="cctype"+str(i)+str(j))
        op = [None]*nrows
        if cctype == "Expiry":
            for i in range(nrows):
                op[i]=fake.credit_card_expire()
        if cctype == "CVC":
            for i in range(nrows):
                op[i]=fake.credit_card_security_code()
        if cctype == "Provider":
            for i in range(nrows):
                op[i]=fake.credit_card_provider()
        if cctype == "Full":
            for i in range(nrows):
                op[i]=fake.credit_card_full()
        return op

    if coltype == "List":
        lst = st.text_input(
            label="enter a list of things, comma seperated", value="foo, bar, baz", key="lab"+str(i)+str(j))
        howlst = st.selectbox(
            "how do you want to generate the column", ('random', 'in sequence'), key="howl"+str(i)+str(j))
        lst = lst.split(sep=",")
        llen = len(lst)
        if howlst == "random":
            lop = []
            for i in range(nrows):
                lop.append(lst[random.randint(0, llen-1)])
        elif howlst == "in sequence":
            lop = []
            for i in range(nrows):
                lop.append(lst[i % llen])

        return lop

    if coltype == "Distribution":
        disttype=st.selectbox("Distribution Type",("Normal Distribution","Exponential Distribution","Visual Distribution"))
        if disttype=="Visual Distribution":
            xmin = st.text_input("Enter lowest x value", value=0, key="xmn"+str(i)+str(j))
            xmax = st.text_input("Enter highest x value", value=1, key="xmx"+str(i)+str(j))
            ymin = st.text_input("Enter lowest y value", value=0, key="ymn"+str(i)+str(j))
            ymax = st.text_input("Enter highest y value", value=1, key="ymx"+str(i)+str(j))
            if st.button("Graph drawing Tool"):
                subprocess.run(["python", "tink3.py",xmin,xmax,ymin,ymax])

            return [None]*nrows
        
        if disttype=="Normal Distribution":
            mean=int(st.text_input(label="enter mean of distribution", value=0, key="ndm"+str(i)+str(j)))
            scale=int(st.text_input(label="enter scale/std.dev of distribution", value=0, key="nds"+str(i)+str(j)))
            return norm.rvs(size=nrows,loc=mean,scale=scale)

        if disttype=="Exponential Distribution":
            mean=int(st.text_input(label="enter mean of distribution", value=0, key="xdm"+str(i)+str(j)))
            scale=int(st.text_input(label="enter scale/std.dev of distribution", value=0, key="xds"+str(i)+str(j)))
            return expon.rvs(size=nrows,loc=mean,scale=scale)

    if coltype == "Python Expression":
        def getcol(colname):
            return tblcols[colnames.index(colname)]
        
        op=[None]*nrows
        st.caption("write a python code or expression to describe the individual element of the column. the code needs to modify the value of op[i] , which is a value of a single element of the column, and i is the row number.")
        st.caption("the function getcol('colname') allows you to access values of other columns to create relationships, eg:")
        st.code("""
            if (getcol('Gender')[i]=='Male'):
                op[i]=fake.first_name_male
            else:
                op[i]=fake.first_name_female   """ )
        content = st_ace(language="python", theme="twilight", auto_update=True,
                         wrap=True, min_lines=1, max_lines=2, key="code"+str(i)+str(j))
        if st.button("save"):
            with open('pyexpr'+f'{i}'+'.py', "w") as myfile:
                myfile.write(content)
            myfile.close()
            for i in range(nrows):
                exec(open('pyexpr'+f'{i}'+'.py').read(),globals(),locals())
            return op



def page_home():
    htmlp1 = '''
            <style>
                .text {
                    color: #000000;
                    -webkit-text-stroke: 0.2px white;
                    text-shadow: 1px 0px 1px #CCCCCC, 0px 1px 1px #EEEEEE, 2px 1px 1px #CCCCCC, 1px 2px 1px #EEEEEE, 3px 2px 1px #CCCCCC, 2px 3px 1px #EEEEEE, 4px 3px 1px #CCCCCC, 3px 4px 1px #EEEEEE, 5px 4px 1px #CCCCCC, 4px 5px 1px #EEEEEE, 6px 5px 1px #CCCCCC, 5px 6px 1px #EEEEEE, 7px 6px 1px #CCCCCC;

                }
                    del {
                    background: #000;
                    color: #fff;
                    text-decoration:none;
                    }
                .bruh{
                    display:inline;
                    margin-right:10px;
                }

            </style>
            <h1 class="text">GENETHOS🧊</h1>
            <h3 class="text bruh"><i>Synthetic Data & Bias Tools</i></h3>
            <!-- <del>v0.0.1</del> -->
            '''

    st.markdown(htmlp1, unsafe_allow_html=True)
    st.subheader("About our App:")
    st.write("The app is divided into three sections, New Data, More Data, and Bias Detection and Mitigation. On the sidebar on the left you can upload a dataset to generate more columns with synthetic data tools or to detect and eliminate bias with Bias tools. The new data ")
    st.write("Our Web-based Synthetic data & Bias tools help you to generate more data or create entirely new data and detect and eliminate bias in datasets.")
    st.subheader("Synthetic Data & it's benefits:")
    st.write('1. Overcoming real data usage restrictions: Real data may have usage constraints due to privacy rules or other regulations. Synthetic data can replicate all important statistical properties of real data without exposing real data, thereby eliminating the issue.')
    st.write('2. Creating data to simulate not yet encountered conditions: Where real data does not exist, synthetic data is the only solution.')
    st.write('3. Immunity to some common statistical problems: These can include item nonresponse, skip patterns, and other logical constraints.')
    st.write('4. Immunity to some common statistical problems: These can include item nonresponse, skip patterns, and other logical constraints.')
    st.subheader('Bias Mitigation & Detection:')
    st.write("For Bias detection our applications utilizes already established metrics and mitigation algorithms by IBM-AIF360. Further In our work we implement those on new datasets. Also we use this tool to interpret if AI models generate bias data. If yes we provide a mitigated data")
