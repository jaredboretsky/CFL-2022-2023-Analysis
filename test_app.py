#!/usr/bin/env python
# coding: utf-8

# In[1]:


from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import base64
import random


# In[10]:


# Initialize Dash app
app = Dash(__name__)

# Define callback to update the scatter plot based on selected week
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('week-dropdown-scatter', 'value'),
     Input('radio-scatter', 'value')])

def update_scatter_plot(selected_week, selected_metric):
    
    # Choose the plot based on the selected metric
    if selected_metric == 'Points':
        # Define the path to the CSV file for the selected week
        csv_path = f"/Users/jaredboretsky/Documents/concordia-bootcamps/ds-final_project/CFL_Data/week_{selected_week}_2023/scoring_breakdown.csv"
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
        csv_path_team = f"/Users/jaredboretsky/Documents/concordia-bootcamps/ds-final_project/CFL_Data/week_{selected_week}_2023/net_offence.csv"
        csv_path_opp = f"//Users/jaredboretsky/Documents/concordia-bootcamps/ds-final_project/CFL_Data/week_{selected_week}_2023/opponent_net_offence.csv"
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

@app.callback(
    Output('first-down', 'figure'),
    [Input('week-dropdown-first-down', 'value')])
    
def update_chart(week_number):
    # Load data for the selected week
    filename = f'/Users/jaredboretsky/Documents/concordia-bootcamps/ds-final_project/CFL_Data/week_{week_number}_2023/first_down_offence.csv'
    df = pd.read_csv(filename)
    filename_standings = f'/Users/jaredboretsky/Documents/concordia-bootcamps/ds-final_project/CFL_Data/week_{week_number}_2023/game_stat_trends.csv'
    df_standings = pd.read_csv(filename_standings)
    # Perform the same data processing as before
    df['1st_down_pass_calls'] = pd.to_numeric(df['1st_down_pass_calls'], errors='coerce')
    df['1st_down_plays'] = pd.to_numeric(df['1st_down_plays'], errors='coerce')
    # Replace '#DIV/0!' with NaN, then fill NaN with 0 (or another appropriate value)
    df.replace('#DIV/0!', pd.NA, inplace=True)
    df.fillna(0, inplace=True)
    # Now perform the division
    df['pass_percentage'] = df['1st_down_pass_calls'] / df['1st_down_plays']
    # Convert other necessary columns
    df['1st_down_avg_yds'] = pd.to_numeric(df['1st_down_avg_yds'])
    if 'Team' in df.columns:
        df['TM'] = df['Team']
    df['TM'] = df['TM'].astype(str)
    
    # Bar chart creation code (same as before)
    data = []
    bar_width = 0.4
    first_pass_bar = True
    first_run_bar = True
    
    for i, team in df.iterrows():
        pass_percentage = team['pass_percentage']
        run_percentage = 1 - pass_percentage
        average_yard = team['1st_down_avg_yds']

        # Pass portion of the bar
        if first_pass_bar:
            data.append(go.Bar(
                x=[team['TM']],
                y=[pass_percentage * average_yard],
                name='Pass',
                width=bar_width,
                marker=dict(color='green'),
                hovertemplate=f"Pass: {pass_percentage:.1%}<br>Avg Yards: {average_yard:.1f}<extra></extra>"
            ))
            first_pass_bar = False
        else:
            data.append(go.Bar(
                x=[team['TM']],
                y=[pass_percentage * average_yard],
                width=bar_width,
                marker=dict(color='green'),
                showlegend=False,
                hovertemplate=f"Pass: {pass_percentage:.1%}<br>Avg Yards: {average_yard:.1f}<extra></extra>"
            ))

    # Run portion of the bar
        if first_run_bar:
            data.append(go.Bar(
                x=[team['TM']],
                y=[run_percentage * average_yard],
                name='Run',
                width=bar_width,
                marker=dict(color='red'),
                base=[pass_percentage * average_yard],
                hovertemplate=f"Run: {run_percentage:.1%}<br>Avg Yards: {average_yard:.1f}<extra></extra>"
            ))
            first_run_bar = False
        else:
            data.append(go.Bar(
                x=[team['TM']],
                y=[run_percentage * average_yard],
                width=bar_width,
                marker=dict(color='red'),
                base=[pass_percentage * average_yard],
                showlegend=False,
                hovertemplate=f"Run: {run_percentage:.1%}<br>Avg Yards: {average_yard:.1f}<extra></extra>"
        ))
    layout = go.Layout(
    barmode='stack',
    title='Average Yards on 1st Down by Team',
    xaxis=dict(title='Teams'),
    yaxis=dict(title='Average Yards on 1st Down'),
    legend=dict(
        x=1,  # Position the legend outside the plot area on the right
        y=1,     # Align the legend with the top of the plot
        bordercolor="Black",
        borderwidth=2
    )
            )

    return {'data': data, 'layout': layout}
# Define the layout of the app
app.layout = html.Div([
    dcc.Dropdown(
        id='week-dropdown-scatter',
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
                       id='radio-scatter'
    ),
    dcc.Dropdown(
        id='week-dropdown-first-down',  # Unique ID for the second chart dropdown
        options=[{'label': f'Week {i}', 'value': i} for i in range(1, 22)],  # Assuming 21 weeks
        value=1  # Default value
    ),
    
    # Second chart (scatter or line chart)
    dcc.Graph(id='first-down')
    
    ])



if __name__ == '__main__':
    app.run_server(debug=True)

