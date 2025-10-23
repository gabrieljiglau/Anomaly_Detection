import sys, os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import importlib
import plotly.express as px

with open('../models/posteriors.pkl', 'rb') as f:
    saved_posteriors = pickle.load(f)
    niw_posteriors = saved_posteriors['niw_posteriors']
    alpha_posteriors = saved_posteriors['alpha_posteriors']
    sticks = saved_posteriors['sticks_list']
    log_likelihoods = saved_posteriors['log_likelihoods']
    active_clusters = saved_posteriors['active_clusters']

iterations = np.arange(1, len(log_likelihoods) + 1)
df_convergence = pd.DataFrame({
    "Iteration": iterations,
    "Log_likelihood": log_likelihoods
})

st.write("Checking convergence of the log likelihood. It should be increasing")

fig = px.line(df_convergence, x="Iteration", y="Log_likelihood",
              markers=True, title="Log likelihoods over iterations")
fig.update_layout(template="plotly_white")
st.plotly_chart(fig, use_container_width=True)
