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
    'Aberdeen', 'Aberdeenshire', 'Abu Dhabi', 'Afghanistan', 'Angus', 'Argyll', 
    'Australia', 'Avon', 'Ayrshire', 'Bahrain', 'Banffshire', 'Barbados', 'Bedfordshire', 
    'Belgium', 'Berkshire', 'Berwickshire', 'Birmingham', 'Brazil', 'Bristol', 'Buckinghamshire', 
    'Caithness', 'Cambridgeshire', 'Canada', 'Cardiff', 'Central London', 'Channel Islands', 'Cheshire',
    'China', 'City of London', 'Clackmannanshire', 'Cleveland', 'Clwyd', 'Cornwall', 'County Antrim', 
    'County Armagh', 'County Derry', 'County Down', 'County Durham', 'County Tyrone', 'Cumbria', 
    'Czech Republic', 'Denmark', 'Derbyshire', 'Devon', 'Dorset', 'Dubai', 'Dumfriesshire', 'Dunbartonshire', 
    'Dyfed', 'East Anglia', 'East London', 'East Lothian', 'East Midlands', 'East Sussex', 'Edinburgh', 
    'Egypt', 'England', 'Essex', 'Fife', 'France', 'Germany', 'Gibraltar', 'Glasgow', 'Gloucestershire', 
    'Greece', 'Gwent', 'Gwynedd', 'Hampshire', 'Herefordshire', 'Hertfordshire', 'Hong Kong', 'Hungary', 
    'Indonesia', 'Inverness-Shire', 'Inverness-shire', 'Iraq', 'Ireland', 'Isle Of Wight', 'Isle of Orkney', 
    'Isle of Wight', 'Italy', 'Kent', 'Kincardineshire', 'Kirkcudbrightshire', 'Lanarkshire', 'Lancashire', 
    'Leicestershire', 'Liberia', 'Lincolnshire', 'London', 'Malta', 'Manchester', 'Merseyside', 'Mid Glamorgan', 
    'Middlesex', 'Midlothian', 'Morayshire', 'Netherlands', 'New Zealand', 'Norfolk', 'North East England', 
    'North Humberside', 'North London', 'North West England', 'North West London', 'North Yorkshire', 
    'Northamptonshire', 'Northern Ireland', 'Northumberland', 'Norway', 'Nottinghamshire', 'Other', 
    'Oxfordshire', 'Perthshire', 'Poland', 'Portugal', 'Powys', 'Renfrewshire', 'Rest of the World', 
    'Ross-Shire', 'Ross-shire', 'Roxburghshire', 'Russian Federation', 'Rutland', 'Saudi Arabia', 'Scotland', 
    'Selkirkshire', 'Shetland Islands', 'Shropshire', 'Singapore', 'Somerset', 'South East England', 
    'South East London', 'South Glamorgan', 'South Humberside', 'South West England', 'South West London', 
    'South Yorkshire', 'Spain', 'St. Vincent & The Grenadines', 'Staffordshire', 'Stirlingshire', 'Suffolk', 
    'Surrey', 'Sutherland', 'Sweden', 'Switzerland', 'Taiwan', 'Thailand', 'Tyne And Wear', 'Tyne and Wear', 
    'USA', 'United Arab Emirates', 'Wales', 'Warwickshire', 'West Glamorgan', 'West London', 'West Lothian', 
    'West Midlands', 'West Midlands (Region)', 'West Sussex', 'West Yorkshire', 'Wigtownshire', 'Wiltshire', 
    'Worcestershire', 'Wrexham', 'Yorkshire and Humberside']


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

st.text('If you are looking to post a job, you came to the right place, we will help you to have\n'
'an estimate of how much the industry is paying for this kind of position.')

cat = st.selectbox('Select a category (just one please)', categories)
if len(cat)>0:
    st.text(cat)

contract_type = st.selectbox('Select a contract type (just one please)', contracts)
if len(contract_type)>0:
    st.text(contract_type)

d_work = st.multiselect('Full time or part time? (can be both)', d_works)
if len(d_work)>0:
    st.text(d_work)

description = st.text_area('Describe the job (minimum 50 characters)')
if len(cat)>0 and len(description)>50:
    cluster = txt_to_cluster([description.lower()], cat[0])
    st.text(cluster)

state = st.selectbox('Select a state? (just one please)', states)
if len(state)>0:
    st.text(state)