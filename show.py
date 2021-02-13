import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import normalize

categories = [
    'accountancy',
    'accountancy_qualified',
    'admin_secretarial_pa',
    'apprenticeships',
    'banking',
    'catering',
    'charity',
    'construction_property',
    'customer_service',
    'education',
    'energy',
    'engineering',
    'estate_agent',
    'factory',
    'finance',
    'fmcg',
    'general_insurance',
    'graduate_training_internships',
    'health',
    'hr',
    'it',
    'law',
    'leisure_tourism',
    'logistics',
    'marketing',
    'media_digital_creative',
    'motoring_automotive',
    'other',
    'purchasing',
    'recruitment_consultancy',
    'retail',
    'sales',
    'science',
    'security_safety',
    'social_care',
    'strategy_consultancy',
    'training',
]

contracts = [
    'Contract',
    'Permanent',
    'Temporary'
]

d_works = [
    'full time',
    'part time'
]

states = [
    'Aberdeen',
    'Aberdeenshire',
    'Abu Dhabi',
    'Angus',
    'Argyll',
    'Australia',
    'Avon',
    'Ayrshire',
    'Bahrain',
    'Banffshire',
    'Barbados',
    'Bedfordshire',
    'Belgium',
    'Berkshire',
    'Berwickshire',
    'Birmingham',
    'Brazil',
    'Bristol',
    'Buckinghamshire',
    'Caithness',
    'Cambridgeshire',
    'Cardiff',
    'Central London',
    'Channel Islands',
    'Cheshire',
    'China',
    'City of London',
    'Clackmannanshire',
    'Cleveland',
    'Clwyd',
    'Cornwall',
    'County Antrim',
    'County Armagh',
    'County Derry',
    'County Down',
    'County Durham',
    'County Tyrone',
    'Cumbria',
    'Derbyshire',
    'Devon',
    'Dorset',
    'Dubai',
    'Dumfriesshire',
    'Dunbartonshire',
    'Dyfed',
    'East Anglia',
    'East London',
    'East Lothian',
    'East Midlands',
    'East Sussex',
    'Edinburgh',
    'Egypt',
    'England',
    'Essex',
    'Fife',
    'France',
    'Germany',
    'Gibraltar',
    'Glasgow',
    'Gloucestershire',
    'Gwent',
    'Gwynedd',
    'Hampshire',
    'Herefordshire',
    'Hertfordshire',
    'Hong Kong',
    'Indonesia',
    'Inverness-Shire',
    'Inverness-shire',
    'Ireland',
    'Isle of Orkney',
    'Isle of Wight',
    'Italy',
    'Kent',
    'Kincardineshire',
    'Lanarkshire',
    'Lancashire',
    'Leicestershire',
    'Liberia',
    'Lincolnshire',
    'London',
    'Malta',
    'Manchester',
    'Merseyside',
    'Mid Glamorgan',
    'Middlesex',
    'Midlothian',
    'Morayshire',
    'Netherlands',
    'New Zealand',
    'Norfolk',
    'North East England',
    'North Humberside',
    'North London',
    'North West England',
    'North West London',
    'North Yorkshire',
    'Northamptonshire',
    'Northern Ireland',
    'Northumberland',
    'Norway',
    'Nottinghamshire',
    'Other',
    'Oxfordshire',
    'Perthshire',
    'Poland',
    'Powys',
    'Renfrewshire',
    'Rest of the World',
    'Ross-Shire',
    'Ross-shire',
    'Roxburghshire',
    'Rutland',
    'Saudi Arabia',
    'Scotland',
    'Selkirkshire',
    'Shetland Islands',
    'Shropshire',
    'Singapore',
    'Somerset',
    'South East England',
    'South East London',
    'South Glamorgan',
    'South Humberside',
    'South West England',
    'South West London',
    'South Yorkshire',
    'Spain',
    'St. Vincent & The Grenadines',
    'Staffordshire',
    'Stirlingshire',
    'Suffolk',
    'Surrey',
    'Sutherland',
    'Sweden',
    'Switzerland',
    'Thailand',
    'Tyne And Wear',
    'Tyne and Wear',
    'USA',
    'United Arab Emirates',
    'Wales',
    'Warwickshire',
    'West Glamorgan',
    'West London',
    'West Lothian',
    'West Midlands',
    'West Midlands (Region)',
    'West Sussex',
    'West Yorkshire',
    'Wigtownshire',
    'Wiltshire',
    'Worcestershire',
    'Wrexham',
    'Yorkshire and Humberside',
]


tfidf_models = {cat: joblib.load(f'models/{cat}_tfidf.joblib') for cat in categories}
pca_models = {cat: joblib.load(f'models/{cat}_pca.joblib') for cat in categories}
kmeans_models = {cat: joblib.load(f'models/{cat}_kmeans.joblib') for cat in categories}

def txt_to_cluster(txt, cat):
    tfidf_x = tfidf_models[cat].transform(txt)
    norm_x = normalize(tfidf_x).toarray()
    pca_x = pca_models[cat].transform(norm_x)
    cluster = kmeans_models[cat].predict(pca_x)
    return cluster[0]


max_rf_regressor = joblib.load('models/max_salary_rf')
min_rf_regressor = joblib.load('models/min_salary_rf')

st.header('If you are looking to post a job, you came to the right place, we will help you to have an estimate of how much the industry is paying for this kind of position.')

cat = st.selectbox('Select a category', categories)

contract_type = st.selectbox('Select a contract type', contracts)

d_work = st.multiselect('Full time or part time?', d_works)

description = st.text_area('Describe the job')

state = st.selectbox('Select a state?', states)

has_category = len(cat)>0
has_contract_type = len(contract_type)>0
has_d_work = len(d_work)>0
has_description = len(cat)>0 and len(description)>50
has_state = len(state)>0

if (
    has_category
    and has_contract_type
    and has_d_work
    and has_description
    and has_description
    and has_state
):
    category_arr = [0]*len(categories)
    category_arr[categories.index(cat)] = 1
    contract_type_arr = [0]*len(contracts)
    contract_type_arr[contracts.index(contract_type)] = 1
    if len(d_work)==2:
        d_work_arr = [0]*len(d_works)
    else:
        d_work_arr = [0]*len(d_works)
        d_work_arr[d_works.index(d_work[0])] = 1
    cluster = txt_to_cluster([description.lower()], cat)
    description_arr = [0]*10
    description_arr[cluster] = 1
    state_arr = [0]*len(states)
    state_arr[states.index(state)] = 1

    final_x = [
        *contract_type_arr,
        *d_work_arr,
        *state_arr,
        *category_arr,
        *description_arr
    ]
    # st.text(f'{len(contract_type_arr), len(d_work_arr), len(state_arr), len(category_arr), len(description_arr)}')

    min_sal = min_rf_regressor.predict([np.array(final_x)])[0]
    max_sal = max_rf_regressor.predict([np.array(final_x)])[0]
    st.header(f"The recoomended salary per hour is minimum: £{min_sal:.2f}, maximum: £{max_sal:.2f}")
    st.header(f"The recoomended salary per day is minimum: £{(min_sal*8):.2f}, maximum: £{(max_sal*8):.2f}")
    st.header(f"The recoomended salary per year is minimum: £{(min_sal*8*255):.2f}, maximum: £{(max_sal*8*255):.2f}")

