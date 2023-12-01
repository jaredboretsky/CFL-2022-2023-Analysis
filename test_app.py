#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import base64
import random


# In[ ]:


# Initialize Dash app
app = Dash(__name__)
server = app.server
# Define callback to update the scatter plot based on selected week
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('week-dropdown', 'value'),
     Input('my-radio-buttons-final', 'value')])

def update_scatter_plot(selected_week, selected_metric):
    
    # Choose the plot based on the selected metric
    if selected_metric == 'Points':
        # Define the path to the CSV file for the selected week
        csv_path = f"data/week_{selected_week}_2023/scoring_breakdown.csv"
        # Check if the file exists before attempting to read it
        if os.path.exists(csv_path):
            # Read the data from the CSV file for the selected week
            df = pd.read_csv(csv_path)
            df = df.drop(9)
            numeric_columns = ['PF', 'PA']
            for col in numeric_columns:
                if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                    # Handle non-numeric data, e.g., replace with NaN or a default value
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            # If the file doesn't exist, create an empty DataFrame
            df = pd.DataFrame(columns=['Team', 'PF', 'PA'])  # Modify columns as needed
        df['Point Differential'] = df['PF'] - df['PA']
        # Rest of the code to create and customize the scatter plot
        padding = 5  # Adjust this padding value as needed
        x_min, x_max = df['PF'].min() - padding, df['PF'].max() + padding
        y_min, y_max = df['PA'].min() - padding, df['PA'].max() + padding

        fig_points = px.scatter(
            df, 
            x='PF', 
            y='PA', 
            text='Team', 
            color='Team',
            labels={'PF': 'Points For', 'PA': 'Points Against', 'Team': 'Team'}, 
            hover_data={'Point Differential': True} 
        )

# Customize the hover template to include the 'Yard Differential'
        fig_points.update_traces(
            hovertemplate='<b>Team:</b> %{text}<br><b>Points For:</b> %{x}<br><b>Points Against:</b> %{y}<br><b>Point Differential:</b> %{customdata[0]}<extra></extra>'
        )
#     # Add labels for each quarter
#     fig.add_annotation(text='Everything is happening', x=x_max-5, y=y_max-5, showarrow=False)
#     fig.add_annotation(text='Things are going bad', x=x_min+5, y=y_max-5, showarrow=False)
#     fig.add_annotation(text='Boring football', x=x_min+5, y=y_min+5, showarrow=False)
#     fig.add_annotation(text='Where you want to be', x=x_max-5, y=y_min+5, showarrow=False)

        fig_points.update_layout(
            title=f'CFL: Points For vs Points Against (Week {selected_week})',
            xaxis_title='Points For',
            yaxis_title='Points Against',
            showlegend=True,
            legend_title_text='Team',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )

        return fig_points
    else:
        csv_path_team = f"data/week_{selected_week}_2023/net_offence.csv"
        csv_path_opp = f"data/week_{selected_week}_2023/opponent_net_offence.csv"
    # Check if the file exists before attempting to read it
        if os.path.exists(csv_path_team):
            # Read the data from the CSV file for the selected week
            df_team = pd.read_csv(csv_path_team)
            df_team = df_team.drop(9)
            df_team.rename(columns={'Yards':'Yards For'}, inplace=True)
            df_team['Yards For'] = df_team['Yards For'].str.replace(r'[^0-9.]', '', regex=True)
            df_team['Yards For'] = pd.to_numeric(df_team['Yards For'], errors='coerce')
        else:
            # If the file doesn't exist, create an empty DataFrame
            df_team = pd.DataFrame(columns=['Team', 'Yards For'])  # Modify columns as needed
            
        if os.path.exists(csv_path_opp):
            # Read the data from the CSV file for the selected week
            df_opp = pd.read_csv(csv_path_opp)
            df_opp = df_opp.drop(9)
            df_opp.rename(columns={'Yards':'Yards Against'}, inplace=True)
            df_opp['Yards Against'] = df_opp['Yards Against'].str.replace(r'[^0-9.]', '', regex=True)
            df_opp['Yards Against'] = pd.to_numeric(df_opp['Yards Against'], errors='coerce')
        else:
            # If the file doesn't exist, create an empty DataFrame
            df_opp = pd.DataFrame(columns=['Team', 'Yards For'])
        
        df = pd.merge(df_team, df_opp, on='Team', how='inner')
        df['Yard Differential'] = df['Yards For'] - df['Yards Against']      
        padding = 50  # Adjust this padding value as needed
        x_min, x_max = df['Yards For'].min() - padding, df['Yards For'].max() + padding
        y_min, y_max = df['Yards Against'].min() - padding, df['Yards Against'].max() + padding

        fig_yards = px.scatter(
            df, 
            x='Yards For', 
            y='Yards Against', 
            text='Team', 
            color='Team',
            labels={'Yards For': 'Yards For', 'Yards Against': 'Yards Against', 'Team': 'Team'}, 
            hover_data={'Yard Differential': True} 
        )

# Customize the hover template to include the 'Yard Differential'
        fig_yards.update_traces(
            hovertemplate='<b>Team:</b> %{text}<br><b>Yards For:</b> %{x}<br><b>Yards Against:</b> %{y}<br><b>Yard Differential:</b> %{customdata[0]}<extra></extra>'
        )
        
#     # Add labels for each quarter
#     fig.add_annotation(text='Everything is happening', x=x_max-5, y=y_max-5, showarrow=False)
#     fig.add_annotation(text='Things are going bad', x=x_min+5, y=y_max-5, showarrow=False)
#     fig.add_annotation(text='Boring football', x=x_min+5, y=y_min+5, showarrow=False)
#     fig.add_annotation(text='Where you want to be', x=x_max-5, y=y_min+5, showarrow=False)

        fig_yards.update_layout(
            title=f'CFL: Yards For vs Yards Against (Week {selected_week})',
            xaxis_title='Yards For',
            yaxis_title='Yards Against',
            showlegend=True,
            legend_title_text='Team',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )
        return fig_yards
# Define the layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='week-dropdown',
        options=[
            {'label': f'Week {week}', 'value': week}
            for week in range(1, 22)  # Assuming you have 17 weeks of data
        ],
        value=1,  # Default to Week 1
        style={'width': '50%'}
    ),
    dcc.Graph(id='scatter-plot'),
    dcc.RadioItems(options=['Points', 'Yards'],
                       value='Yards',
                       inline=True,
                       id='my-radio-buttons-final')
    ])



if __name__ == '__main__':
    app.run_server(debug=True)

