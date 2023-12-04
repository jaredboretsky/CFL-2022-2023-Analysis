#!/usr/bin/env python
# coding: utf-8

# In[2]:


from dash import Dash, html, dash_table, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import os
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import base64
import random


# In[53]:


# Initialize Dash app
app = Dash(__name__)
server = app.server  # This exposes the Flask server for Gunicorn to use
# Define callback to update the scatter plot based on selected week
@app.callback(
    Output('scatter-plot', 'figure'),
    [Input('week-slider-scatter', 'value'),
     Input('radio-scatter', 'value')])

def update_scatter_plot(selected_week, selected_metric):
    
    team_color_map = {
        'BC': 'darkorange',   # Replace 'Team1', 'Team2', etc. with actual team names
        'CGY': 'tomato',
        'EDM': 'lightgreen',
        'HAM': 'yellow',   # Replace 'Team1', 'Team2', etc. with actual team names
        'MTL': 'maroon',
        'OTT': 'black',
        'SSK': 'forestgreen',   # Replace 'Team1', 'Team2', etc. with actual team names
        'TOR': 'royalblue',
        'WPG': 'navy'
        # Add more teams and their colors here
    }
    
    # Choose the plot based on the selected metric
    if selected_metric == 'Points':
        # Define the path to the CSV file for the selected week
        csv_path = f"data/week_{selected_week}_2023/scoring_breakdown.csv"

        # Check if the file exists before attempting to read it
        if os.path.exists(csv_path):
            # Read the data from the CSV file for the selected week
            df = pd.read_csv(csv_path)
            df = df = df.drop(9)
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
        padding = 10  # Adjust this padding value as needed
        x_min, x_max = df['PF'].min() - padding, df['PF'].max() + padding
        y_min, y_max = df['PA'].min() - padding, df['PA'].max() + padding

        fig_points = px.scatter(
            df, 
            x='PF', 
            y='PA',  
            color='Team',
            color_discrete_map=team_color_map,
            labels={'PF': 'Points For', 'PA': 'Points Against', 'Team': 'Team'}, 
            hover_data={'Point Differential': True},
            custom_data=['Team', 'Point Differential']
        )
        logo_dir = 'data/CFL_Logos/'
        Teams = ['BC', 'CGY', 'EDM', 'HAM', 'MTL', 'OTT', 'SSK', 'TOR', 'WPG']
        x_range = df['PF'].max() - df['PF'].min()
        y_range = df['PA'].max() - df['PA'].min()
        # Set a relative size for the images (e.g., 5% of the data range)
        relative_size_x = 0.15 * x_range
        relative_size_y = 0.15 * y_range
        for team in Teams:
            team_logo_path = os.path.join(logo_dir, f'{team}_logo.png')
            if os.path.exists(team_logo_path):  # Check if the file exists
                with open(team_logo_path, 'rb') as image_file:
                    image_encoded = base64.b64encode(image_file.read()).decode()

        # Get the coordinates for the team
                team_data = df[df['Team'] == team]
                if not team_data.empty:  # Check if team data is not empty
                    x_value = team_data['PF'].values[0]
                    y_value = team_data['PA'].values[0]
                    fig_points.add_layout_image(
                        source=f'data:image/png;base64,{image_encoded}',
                        x=x_value,  # X-coordinate based on Points For
                        y=y_value,  # Y-coordinate based on Points Against
                        xref="x",
                        yref="y",
                        xanchor='center',
                        yanchor='middle',
                        sizex=relative_size_x,  # Adjust the size of the logos as needed
                        sizey=relative_size_y,
                        opacity=1,
                        layer="above"
                    )
                else:
                    print(f"No data found for team {team}")
            else:
                print(f"Logo file not found for team {team}")
        
        fig_points.update_traces(
            marker=dict(opacity=0),
            hovertemplate='<b>Team:</b> %{customdata[0]}<br><b>Yards For:</b> %{x}<br><b>Yards Against:</b> %{y}<br><b>Yard Differential:</b> %{customdata[1]}<extra></extra>'
        )
        fig_points.update_layout(
            title=f'CFL: Cumulative Points For vs Cumulative Points Against (Week {selected_week})',
            title_x = 0.5,
            xaxis_title='Points For',
            yaxis_title='Points Against',
            showlegend=False,
            legend_title_text='Team',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )

        return fig_points
    else:
        csv_path_team = f"data/week_{selected_week}_2023/net_offence.csv"
        csv_path_opp = f"/data/week_{selected_week}_2023/opponent_net_offence.csv"
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
        padding = 100  # Adjust this padding value as needed
        x_min, x_max = df['Yards For'].min() - padding, df['Yards For'].max() + padding
        y_min, y_max = df['Yards Against'].min() - padding, df['Yards Against'].max() + padding

        fig_yards = px.scatter(
            df, 
            x='Yards For', 
            y='Yards Against',  
            color='Team',
            color_discrete_map=team_color_map,
            labels={'Yards For': 'Yards For', 'Yards Against': 'Yards Against', 'Team': 'Team'}, 
            hover_data={'Yard Differential': True},
            custom_data = ['Team', 'Yard Differential']
        )
        logo_dir = 'data/CFL_Logos/'
        Teams = ['BC', 'CGY', 'EDM', 'HAM', 'MTL', 'OTT', 'SSK', 'TOR', 'WPG']
        x_range = df['Yards For'].max() - df['Yards For'].min()
        y_range = df['Yards Against'].max() - df['Yards Against'].min()
        # Set a relative size for the images (e.g., 5% of the data range)
        relative_size_x = 0.15 * x_range
        relative_size_y = 0.15 * y_range
        for team in Teams:
            team_logo_path = os.path.join(logo_dir, f'{team}_logo.png')
            if os.path.exists(team_logo_path):  # Check if the file exists
                with open(team_logo_path, 'rb') as image_file:
                    image_encoded = base64.b64encode(image_file.read()).decode()

        # Get the coordinates for the team
                team_data = df[df['Team'] == team]
                if not team_data.empty:  # Check if team data is not empty
                    x_value = team_data['Yards For'].values[0]
                    y_value = team_data['Yards Against'].values[0]
                    fig_yards.add_layout_image(
                        source=f'data:image/png;base64,{image_encoded}',
                        x=x_value,  # X-coordinate based on Points For
                        y=y_value,  # Y-coordinate based on Points Against
                        xref="x",
                        yref="y",
                        xanchor='center',
                        yanchor='middle',
                        sizex=relative_size_x,  # Adjust the size of the logos as needed
                        sizey=relative_size_y,
                        opacity=1,
                        layer="above"
                    )
                else:
                    print(f"No data found for team {team}")
            else:
                print(f"Logo file not found for team {team}")
        # Customize the hover template to include the 'Yard Differential'
        fig_yards.update_traces(
            marker=dict(opacity=0),
            hovertemplate='<b>Team:</b> %{customdata[0]}<br><b>Yards For:</b> %{x}<br><b>Yards Against:</b> %{y}<br><b>Yard Differential:</b> %{customdata[1]}<extra></extra>'
        )
        

        fig_yards.update_layout(
            title=f'CFL: Cumulative Yards For vs Cumulative Yards Against (Week {selected_week})',
            title_x=0.5,
            xaxis_title='Yards For',
            yaxis_title='Yards Against',
            showlegend=False,
            legend_title_text='Team',
            xaxis=dict(range=[x_min, x_max]),
            yaxis=dict(range=[y_min, y_max])
        )
        return fig_yards

@app.callback(
    Output('first-down', 'figure'),
    [Input('week-dropdown-first-down', 'value')])
    
def update_first_down(week_number):
    # Load data for the selected week
    filename = f'data/week_{week_number}_2023/first_down_offence.csv'
    df = pd.read_csv(filename)
    df = df.drop(9)
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

@app.callback(
    Output('agression', 'figure'),
    [Input('week-dropdown-agression', 'value')])
    
def update_agression(week_number):
    filename = f'data/agression_data.csv'
    df = pd.read_csv(filename)
    df = df[df['Week'] == week_number]
    df['PF_per_game'] = df['PF']/df['GP']
    # Bar chart creation code (same as before)
    data = []
    bar_width = 0.4
    fig = px.bar(df, x='Team', y=['C2_att', '20+_yds_Att_per_game', 'Att', 'PF_per_game'], barmode='group')
    
    fig.for_each_trace(lambda t: t.update(name = {
        'C2_att': '2-Point Conversion Attempts',
        '20+_yds_Att_per_game': 'Deep Passes (20+ Yards) Attempted Per Game',
        'Att': '3rd Down Conversion Attempts',
        'PF_per_game': 'Points For Per Game'
    }[t.name]))
    
    fig.update_layout(
        title='Agressive Plays and Points For per Team',
        title_x=0.5,
        xaxis_title='Teams',
        yaxis_title='Values',
        legend_title='Legend',
        barmode='group'
    )

    return fig

@app.callback(
    Output('second-down', 'figure'),
    [Input('week-slider-second-down', 'value'),
     Input('heatmap-type', 'value')])
    
def update_second_down(week_number, heatmap_type):
    if heatmap_type == 'offense':
        offense_file_name = f'data/week_{week_number}_2023/second_down_conversions.csv'
        df = pd.read_csv(offense_file_name)
    else:
        defense_file_name = f'data/week_{week_number}_2023/opponent_second_down_conversions.csv'
        df = pd.read_csv(defense_file_name)
    
    # Create two separate dataframes
    df_percentages = df[['Team', '1-3_yds_%', '4-6_yds_%', '7+_yds_%']].set_index('Team')
    df_yards_to_go = df[['Team', 'Yds_to_go_Avg']].set_index('Team')
    df_percentages.columns = ['2nd and Short<br>(1-3 Yards) Conversion Rate', 
                          '2nd and Medium<br>(4-6 Yards) Conversion Rate', 
                          '2nd and Long<br>(7+ Yards) Conversion Rate']
    df_yards_to_go.columns = ['Average Yards<br>To Go on 2nd Down']
    
    # Create two separate heatmaps
    fig = make_subplots(
        rows=1, cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.001,  # Spacing between heatmaps
        column_widths=[0.7, 0.3],  # Adjusts the relative widths of the two heatmaps
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}]]
    )

    # Assuming df_percentages and df_yards_to_go are your two DataFrames
    # Add the percentage heatmap as the first subplot
    
    fig.add_trace(
        go.Heatmap(
            z=df_percentages.values,
            x=df_percentages.columns,
            y=df_percentages.index,
            coloraxis="coloraxis1"
        ),
        row=1, col=1
    )

    # Add the yards to go heatmap as the second subplot
    fig.add_trace(
        go.Heatmap(
            z=df_yards_to_go.values,
            x=df_yards_to_go.columns,
            y=df_yards_to_go.index,
            coloraxis="coloraxis2"
        ),
        row=1, col=2
    )
# Update the layout for the subplot figure
    if heatmap_type == 'offense':
        fig.update_layout(
            coloraxis1=dict(colorscale='Blues', colorbar=dict(title='Percentage', x=-.2)),  # Color scale and legend for 1st heatmap
            coloraxis2=dict(colorscale='Greens', colorbar=dict(title='Yards to Go', x=1), reversescale=True),   # Color scale and legend for 2nd heatmap
            title_text="Team Efficiency on 2nd Down",
            title_x=0.5,
            showlegend=False
        )
    else:
        fig.update_layout(
            coloraxis1=dict(colorscale='Blues', colorbar=dict(title='Percentage', x=-.2), reversescale=True),  # Color scale and legend for 1st heatmap
            coloraxis2=dict(colorscale='Greens', colorbar=dict(title='Yards to Go', x=1)),   # Color scale and legend for 2nd heatmap
            title_text="Opponent Efficiency on 2nd Down",
            title_x=0.5,
            showlegend=False
        )

# Move the column names to the top
    fig.update_xaxes(side="top", row=1, col=1)
    fig.update_xaxes(side="top", row=1, col=2)
    # Update x-axis properties to adjust label angle and alignment
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10), row=1, col=1)
    fig.update_xaxes(tickangle=0, tickfont=dict(size=10), row=1, col=2)

    
# Create the Dash layout using the combined figure
    app.layout = html.Div([
        html.H1("CFL Team Efficiency on 2nd Down"),
        dcc.Graph(
            id='combined_heatmap',
            figure=fig
        )
    ])
    
    return fig

def create_big_play_analysis():
    df = pd.read_csv('data/week_21_2023/big_play_analysis.csv')
    df.drop(9, inplace=True)
    
    df['Offensive_Cumsum'] = df['Total']
    df['Defensive_Cumsum'] = df['Opp_off_20+_Rush'] + df['Opp_off_30+_Pass']+df['Opp_kick_Punt_Rets']+df['Opp_kick_K/O_Rets']+df['Opp_kick_FGM_Ret']
    max_offensive = df[['Offensive_Cumsum']].sum(axis=1).max()
    max_defensive = df[['Defensive_Cumsum']].sum(axis=1).max()
    max_value = max(max_offensive, max_defensive)

    # Offset for creating whitespace in the middle
    offset = 5
    max_tick = max_value

    tick_values_positive = list(range(5, max_tick + 10, 5))
    tick_values_negative = [-tick for tick in tick_values_positive]
    tick_values = tick_values_negative + tick_values_positive

    # Generate tick labels. These will be the text labels that appear at the tick marks.
    # For the positive side, we subtract the offset to start the labels at 0.
    # For the negative side, we add the offset to start the labels at 0.
    tick_labels_negative = [str((tick + 5)*-1) for tick in tick_values_negative]
    tick_labels_positive = [str(tick - 5) for tick in tick_values_positive]
    tick_labels = tick_labels_negative + tick_labels_positive

    fig = go.Figure(
        data=[
            go.Bar(
                name='Passes (30+ Yards)',
                y=df['Team'],
                x=df['30+_Pass'],
                orientation='h',
                base=offset,
                marker={'color': 'red'},
                hovertemplate=[
                    f'Team: {team}<br>Passes: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['30+_Pass'], df['Offensive_Cumsum'])
                ]
            ),
            go.Bar(
                name='Rushes (20+ Yards)',
                y=df['Team'],
                x=df['20+_Rush'],
                orientation='h',
                base = offset+df['30+_Pass'],
                marker={'color': 'blue'},
                hovertemplate=[
                    f'Team: {team}<br>Rushes: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['20+_Rush'], df['Offensive_Cumsum'])
                ]
        ),
            go.Bar(
                name='Punt Returns (30+ Yards)',
                y=df['Team'],
                x=df['Punt_Rets'],
                orientation='h',
                base=df['20+_Rush']+df['30+_Pass']+offset,
                marker={'color': 'Green'},
                hovertemplate=[
                    f'Team: {team}<br>Punt Returns: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['Punt_Rets'], df['Offensive_Cumsum'])
                ]
            ),
            go.Bar(
                name='Kickoff Returns (40+ Yards)',
                y=df['Team'],
                x=df['K/O_Rets'],
                orientation='h',
                base=df['20+_Rush']+df['30+_Pass']+df['Punt_Rets']+offset,
                marker={'color': 'Brown'},
                hovertemplate=[
                    f'Team: {team}<br>Kickoff Returns: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['K/O_Rets'], df['Offensive_Cumsum'])
                ]
            ),
            go.Bar(
                name='Missed Field Goal Returns (40+ Yards)',
                y=df['Team'],
                x=df['FGM_Rets'],
                orientation='h',
                base=df['20+_Rush']+df['30+_Pass']+df['Punt_Rets']+df['K/O_Rets']+offset,
                marker={'color': 'Orange'},
                hovertemplate=[
                    f'Team: {team}<br>Missed Field Goal Returns: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['FGM_Rets'], df['Offensive_Cumsum'])
                ]
            ),
            go.Bar(
                name='Passes (30+ Yards)',
                y=df['Team'],
                x=-df['Opp_off_30+_Pass'],
                orientation='h',
                base=-offset,
                marker={'color': 'red'},
                hovertemplate=[
                    f'Team: {team}<br>Passes: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['Opp_off_30+_Pass'], df['Defensive_Cumsum'])
                ],
                showlegend=False
            ),
            go.Bar(
                name='Rushes (20+ Yards)',
                y=df['Team'],
                x=-df['Opp_off_20+_Rush'],
                orientation='h',
                base = -df['Opp_off_30+_Pass']-offset,
                marker={'color': 'blue'},
                hovertemplate=[
                    f'Team: {team}<br>Rushes: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['Opp_off_20+_Rush'], df['Defensive_Cumsum'])
                ],
                showlegend=False
            ),

            go.Bar(
                name='Punt Returns',
                y=df['Team'],
                x=-df['Opp_kick_Punt_Rets'],
                orientation='h',
                base=-df['Opp_off_20+_Rush']-df['Opp_off_30+_Pass']-offset,
                marker={'color': 'Green'},
                hovertemplate=[
                    f'Team: {team}<br>Punt Returns: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['Opp_kick_Punt_Rets'], df['Defensive_Cumsum'])
                ],
                showlegend=False
            ),
            go.Bar(
                name='Kickoff Returns',
                y=df['Team'],
                x=-df['Opp_kick_K/O_Rets'],
                orientation='h',
                base=-df['Opp_off_20+_Rush']-df['Opp_off_30+_Pass']-df['Opp_kick_Punt_Rets']-offset,
                marker={'color': 'Brown'},
                hovertemplate=[
                    f'Team: {team}<br>Kickoff Returns: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['Opp_kick_K/O_Rets'], df['Defensive_Cumsum'])
                ],
                showlegend=False
            ),
            go.Bar(
                name='Missed Field Goal Returns',
                y=df['Team'],
                x=-df['Opp_kick_FGM_Ret'],
                orientation='h',
                base=-df['Opp_off_20+_Rush']-df['Opp_off_30+_Pass']-df['Opp_kick_Punt_Rets']-df['Opp_kick_K/O_Rets']-offset,
                marker={'color': 'Orange'},
                hovertemplate=[
                    f'Team: {team}<br>Missed Field Goal Returns: {play}<br>Total Big Plays: {total}<extra></extra>' 
                    for team, play, total in zip(df['Team'], df['Opp_kick_FGM_Ret'], df['Defensive_Cumsum'])
                ],
                showlegend=False
            ),
        ]
    )

    fig.update_layout(
        title='Big Play Analysis',
        barmode='overlay',
        xaxis=dict(
            tickvals=tick_values,
            ticktext=tick_labels,
            showline=True,  # To show the y-axis line, which is the center line in this graph
            linewidth=1,
            linecolor='rgba(0, 0, 0, 0.5)',
            zeroline=False,  # Hide the zero line
        ),
        yaxis=dict(
            title='',  # No title for y-axis
            showline=False,  # Show the y-axis line
            showgrid=False,  # Turn off the grid lines
            showticklabels=False
        ),
                # Set the plot's background color to white for the gap effect
        plot_bgcolor='white',
        annotations=[
            # Offense Label
            dict(
                x=max_value / 2,  # Adjust this value as needed
                y=1.1,  # Slightly above the top of the graph
                xref='x',
                yref='paper',
                text='Offense (Big Plays Gained)',
                showarrow=False,
                font=dict(color='black', size=14),
                align='center',
            ),
            # Defense Label
            dict(
                x=-max_value / 2,  # Adjust this value as needed
                y=1.1,  # Slightly above the top of the graph
                xref='x',
                yref='paper',
                text='Defense (Big Plays Allowed)',
                showarrow=False,
                font=dict(color='black', size=14),
                align='center',
            ),
            ] + [
                    # Annotations for team names or other static text
            dict(
                x=0,
                y=idx,
                text=team,
                xref='x',
                yref='y',
                showarrow=False,
                font=dict(color='black', size=12),
                align='center',
                xanchor='center',
                yanchor='middle'
            ) for idx, team in enumerate(df['Team'])
        ]
    )
    return fig
# Define the layout of the app
Teams = ['BC', 'CGY', 'EDM', 'HAM', 'MTL', 'OTT', 'SSK', 'TOR', 'WPG']
@app.callback(
    Output('kicking', 'figure'),
    [Input('team-dropdown-kicking', 'value'),
     Input('kick-type', 'value')]
)

def update_kicking(Team, kick_type):
    IMAGE_FILENAME1 = 'data/CFL_Field.png'
    image1 = base64.b64encode(open(IMAGE_FILENAME1, 'rb').read())
    if kick_type == 'kickoffs':   
        fig_kickoff = go.Figure(
            data=[go.Bar(x=[0, 1, 2], y=[0, 0, 0], showlegend=False)],
            layout=go.Layout(
                title_text="Kickoff Yard Position Analysis",
                title_x=0.5  # Center the title
            )
        )
        fig_kickoff.add_layout_image(
            dict(
                source='data:image/png;base64,{}'.format(image1.decode()),
                xref="x",
                yref="paper",
                x=0, y=1,
                sizex=150,
                sizey=2,
                opacity=0.5,
                layer="below"
            )
        )

        fig_kickoff.update_layout(template="plotly_white")

        file_path_kickoff = 'data/week_21_2023/kickoff_analysis.csv'
        df_kickoff = pd.read_csv(file_path_kickoff)

        if not df_kickoff[df_kickoff['Team'] == Team].empty:
            average_Ydl = df_kickoff[df_kickoff['Team'] == Team]['Oppt_start_Av_YL'].iloc[0]  # Use mean() for aggregation
            yard_start_line = 130 - average_Ydl
            avg_value = df_kickoff[df_kickoff['Team'] == Team]['Regular_K/Os_Avg'].iloc[0]
            returns_40 = df_kickoff[df_kickoff['Team'] == Team]['Oppt_returns_40+'].iloc[0]
        else:
            print("Kickoff didn't work" + Team)
        x_axis_range = [0, 152]  # Adjust min_x and max_x as needed
        y_axis_range = [0, 12]  # Adjust min_y and max_y as needed

        # Adding lines for  opponent yard start line
        fig_kickoff.add_shape(
            type="line",
            x0=yard_start_line, 
            y0=0, 
            x1=yard_start_line, 
            y1=1,
            line=dict(color="RoyalBlue", width=3),
            xref="x", yref="paper",
            name = "Opponent Average Starting Yard Line",
            showlegend=False
        )
        fig_kickoff.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color="RoyalBlue", width=4),
            name="Opponent Average <br> Starting Yard Line"
        ))
        #adding line for average KO length
        fig_kickoff.add_shape(type="line",
            x0= 50,  # Starting at the 30 yard line
            y0= 6,
            y1= 6,
            x1= 40 + (avg_value),  # Ending at average net yards from the 50
            line=dict(color="Green", width=4),
            xref="x", yref="y",
            name = "Average Kickoff Length"
                
        )
        fig_kickoff.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color="Green", width=4),
            name="Average Kickoff Length"
        ))
        #add markers for returns 40+ yards:
        # Add scatter points for returns of 40+ yards
        max_x = 60  # Maximum x-coordinate for scatter points
        min_x = 70  # Minimum x-coordinate for scatter points
        max_y = 9.5# Maximum y-coordinate for scatter points
        min_y = 3.5
        spread_x_factor = (max_x - min_x) / returns_40  # Factor to spread points horizontally
        spread_y_factor = (max_y - min_y) / returns_40  # Factor to spread points vertically

        for i in range(int(returns_40)):
            x_coordinate = min_x + i * spread_x_factor  # Spread points evenly across x-axis
            y_coordinate = min_y + i * spread_y_factor  # Spread points evenly across y-axis
            fig_kickoff.add_trace(go.Scatter(
                x=[x_coordinate],
                y=[y_coordinate],  
                marker=dict(color="Red", size=12),
                mode="markers",
                name="Opponent 40+ Yard Returns",
                showlegend=False
                ))
        #Set fixed axis ranges
        fig_kickoff.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            line=dict(color="Red", width=4),
            name="Opponent 40+ Yard Returns"
        ))
        fig_kickoff.update_xaxes(range=x_axis_range, showticklabels=False)
        fig_kickoff.update_yaxes(range=y_axis_range, showticklabels=False)
        
        return fig_kickoff
    
    else:
        fig_punts = go.Figure(
            data=[go.Bar(x=[0, 1, 2], y=[0, 0, 0], showlegend=False)],
            layout=go.Layout(
                title_text="Punting Yard Position Analysis",
                title_x=0.5  # Center the title
            )
        )        
        fig_punts.add_layout_image(
            dict(
                source='data:image/png;base64,{}'.format(image1.decode()),
                xref="x",
                yref="paper",
                x=0, y=1,
                sizex=150,
                sizey=2,
                opacity=0.5,
                layer="below"
            )
        )

        fig_punts.update_layout(template="plotly_white")
        
        file_path_punting = 'data/week_21_2023/punting_analysis.csv'
        df_punt = pd.read_csv(file_path_punting)
        if not df_punt[df_punt['Team'] == Team].empty:
            adjusted_avg_value = df_punt[df_punt['Team'] == Team]['Adjusted_Avg'].iloc[0]
            inside_10 = df_punt[df_punt['Team'] == Team]['I10'].iloc[0]
            returns_30 = df_punt[df_punt['Team'] == Team]['Oppt_returns_30+'].iloc[0]
            
        else:
            print("punt didn't work" + Team)
        
        x_axis_range = [0, 152]  # Adjust min_x and max_x as needed
        y_axis_range = [0, 12]  # Adjust min_y and max_y as needed
        
        start_x = 60
        end_x = 60 + adjusted_avg_value

        # Add a rectangle to represent the net yardage
        fig_punts.add_shape(type="rect",
            x0=start_x, y0=0,
            x1=end_x, y1=1,
            line=dict(color="Blue"),
            fillcolor="LightBlue",
            opacity=0.3,
            xref="x", yref="paper",
            name = "Adjusted Net Punt Yardage",
            showlegend = False
        )
        fig_punts.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='lines',
            line=dict(color="LightBlue", width=4),
            name="Adjusted Net Punt Yardage <br> (Visualized from the 40)"
        ))
        max_x_inside_10 = 130  # Maximum x-coordinate for scatter points
        min_x_inside_10 = 120  # Minimum x-coordinate for scatter points
        max_y = 10# Maximum y-coordinate for scatter points
        min_y = 2.5
        spread_x_factor_inside_10 = (max_x_inside_10 - min_x_inside_10) / inside_10  # Factor to spread points horizontally
        spread_y_factor_inside_10 = (max_y - min_y) / inside_10  # Factor to spread points vertically

        for i in range(int(inside_10)):
            x_coordinate = min_x_inside_10 + i * spread_x_factor_inside_10  # Spread points evenly across x-axis
            y_coordinate = min_y + i * spread_y_factor_inside_10  # Spread points evenly across y-axis
            fig_punts.add_trace(go.Scatter(
                x=[x_coordinate],
                y=[y_coordinate],  
                marker=dict(color="Green", size=12),
                mode="markers",
                name="Punts Inside Opponent's 10",
                showlegend=False
                ))
        fig_punts.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            line=dict(color="Green", width=4),
            name="Punts Inside Opponent's <br> 10 Yard Line"
        ))
        min_x_returns_30 = 55  # Maximum x-coordinate for scatter points
        max_x_returns_30 = 45  # Minimum x-coordinate for scatter points
        spread_x_factor_returns_30 = (max_x_returns_30 - min_x_returns_30) / returns_30  # Factor to spread points horizontally
        spread_y_factor_returns_30 = (max_y - min_y) / returns_30  # Factor to spread points vertically

        for i in range(int(returns_30)):
            x_coordinate = min_x_returns_30 + i * spread_x_factor_returns_30  # Spread points evenly across x-axis
            y_coordinate = min_y + i * spread_y_factor_returns_30  # Spread points evenly across y-axis
            fig_punts.add_trace(go.Scatter(
                x=[x_coordinate],
                y=[y_coordinate],  
                marker=dict(color="Red", size=12),
                mode="markers",
                name="Opponent 30+ Yard Returns",
                showlegend=False
                ))
        fig_punts.add_trace(go.Scatter(
            x=[None],
            y=[None],
            mode='markers',
            line=dict(color="Red", width=4),
            name="Opponent 30+ Yard Returns"
        ))
        #Set fixed axis ranges
        fig_punts.update_xaxes(range=x_axis_range, showticklabels=False)
        fig_punts.update_yaxes(range=y_axis_range, showticklabels=False)
    
        return fig_punts
    
@app.callback(
    Output('rushing-passing', 'figure'),
    [Input('offense-defense-selector-rushing-passing', 'value'),
     Input('rushing-passing-selector', 'value')
    ]
)
def update_rushing_passing(off_def, rush_pass):  
    
    custom_colors = ['darkorange', 'tomato', 'lightgreen', 'yellow', 'maroon', 'black', 'forestgreen', 'royalblue', 'navy']
    
    dfs = []
    for week in range(1, 22):  # Adjust the range based on your week numbers
        file_path = f'data/week_{week}_2023/rushing_analysis.csv'
        week_df = pd.read_csv(file_path)
        week_df['Week'] = week
        dfs.append(week_df)
    # Concatenate all DataFrames into one
    combined_df_rushing = pd.concat(dfs, ignore_index=True)
    combined_df_rushing = combined_df_rushing[combined_df_rushing['Team'] != 'CFL']

    dfs = []
    for week in range(1, 22):  # Adjust the range based on your week numbers
        file_path = f'data/week_{week}_2023/passing_analysis_base_data.csv'
        week_df = pd.read_csv(file_path)
        week_df['Week'] = week
        dfs.append(week_df)
    # Concatenate all DataFrames into one
    combined_df_passing = pd.concat(dfs, ignore_index=True)
    combined_df_passing = combined_df_passing[combined_df_passing['Team'] != 'CFL']

    dfs = []
    for week in range(1, 22):  # Adjust the range based on your week numbers
        file_path = f'data/week_{week}_2023/opponent_passing_analysis_base_data.csv'
        week_df = pd.read_csv(file_path)
        week_df['Week'] = week
        dfs.append(week_df)
    # Concatenate all DataFrames into one
    combined_df_opponent_passing = pd.concat(dfs, ignore_index=True)
    combined_df_opponent_passing = combined_df_opponent_passing[combined_df_opponent_passing['Team'] != 'CFL']



    # Function to clean and convert 'Yards' column to numeric
    def clean_and_convert(df, column_name):
        # Inspect unique values (optional step for troubleshooting)
        # print(df[column_name].unique())

        # Replace any non-numeric characters (if any) and convert to float
        df[column_name] = df[column_name].replace('[^0-9.]', '', regex=True).astype(float)

        # Fill NaNs with 0 (if appropriate)
        df[column_name].fillna(0, inplace=True)

        return df

    # Apply the function to each DataFrame
    combined_df_rushing = clean_and_convert(combined_df_rushing, 'Yards')
    combined_df_rushing = clean_and_convert(combined_df_rushing, 'Opp_rush_Yards')
    combined_df_passing = clean_and_convert(combined_df_passing, 'Yards')
    combined_df_opponent_passing = clean_and_convert(combined_df_opponent_passing, 'Yards')


    pivot_rush_data_offense = combined_df_rushing.pivot_table(
        index='Week', 
        columns='Team', 
        values='Yards', 
        aggfunc='sum'
    ).fillna(0)

    pivot_rush_data_defense = combined_df_rushing.pivot_table(
        index='Week', 
        columns='Team', 
        values='Opp_rush_Yards', 
        aggfunc='sum'
    ).fillna(0)

    pivot_pass_data_offense = combined_df_passing.pivot_table(
        index='Week', 
        columns='Team', 
        values='Yards', 
        aggfunc='sum'
    ).fillna(0)

    pivot_pass_data_defense = combined_df_opponent_passing.pivot_table(
        index='Week', 
        columns='Team', 
        values='Yards', 
        aggfunc='sum'
    ).fillna(0)
    
    if rush_pass == 'Rushing':
        if off_def == 'Offense':
            data = pivot_rush_data_offense
        else:
            data = pivot_rush_data_defense
    else:
        if off_def == 'Offense':
            data = pivot_pass_data_offense
        else:
            data = pivot_pass_data_defense
        # Create and return the figure
    fig = go.Figure()
    for i, team in enumerate(data.columns):
        fig.add_trace(go.Scatter(
            x=data.index, 
            y=data[team], 
            mode='lines+markers',
            name=team,
            line=dict(color=custom_colors[i % len(custom_colors)])
        )
    )
    if rush_pass == 'Rushing':
        if off_def == 'Offense':
            fig.update_layout(
                title='Cumulative Rushing Yards by Team Over the Season',
                title_x=0.5,
                xaxis_title='Week',
                yaxis_title='Cumulative Rushing Yards',
                hovermode='x unified',
                width=850,
                height=600
            )
        else:
            fig.update_layout(
                title='Cumulative Rushing Yards Against by Team Over the Season',
                title_x=0.5,
                xaxis_title='Week',
                yaxis_title='Cumulative Rushing Yards',
                hovermode='x unified',
                width=850,
                height=600
            )
    else:
        if off_def == 'Offense':
            fig.update_layout(
                title='Cumulative Passing Yards by Team Over the Season',
                title_x=0.5,
                xaxis_title='Week',
                yaxis_title='Cumulative Passing Yards',
                hovermode='x unified',
                width=850,
                height=600
            )
        else:
            fig.update_layout(
            title='Cumulative Passing Yards Against by Team Over the Season',
            title_x=0.5,
            xaxis_title='Week',
            yaxis_title='Cumulative Passing Yards',
            hovermode='x unified',
            width=850,
            height=600
        )

    return fig

@app.callback(
    Output('passing-range', 'figure'),
    [Input('offense-defense-selector-passing-range', 'value')]
)

def update_graph(off_def):
    if off_def == 'Offense':
        filtered_df = pd.read_csv('data/week_21_2023/passing_analysis_range_data.csv')
        filtered_df = filtered_df.drop(9)
        filtered_df['0-9_yds_Att'] = pd.to_numeric(filtered_df['0-9_yds_Att'], errors='coerce')
        filtered_df['total_effic'] = [99.5, 83.8, 89.5, 87.0, 95.2, 82.1, 89.4, 104.1, 116.2]

    else:
        filtered_df = pd.read_csv('data/week_20_2023/opponent_passing_analysis_range_data.csv')
        filtered_df = filtered_df.drop(9)
        filtered_df['0-9_yds_Att'] = pd.to_numeric(filtered_df['0-9_yds_Att'], errors='coerce')
        filtered_df['total_effic'] = [89.9, 92.9, 104.0, 94.2, 83.4, 101.8, 105.6, 95.2, 81.7]
        
    
    # Normalizing attempt counts within each pass type
    pass_types = ['0-9_yds_Att', '10-19_yds_Att', '20+_yds_Att']
    for pass_type in pass_types:
        max_attempts = filtered_df[pass_type].max()
        filtered_df[f'{pass_type}_norm'] = (filtered_df[pass_type] / max_attempts) * 100
    
    # Create traces for each range with normalized bubble sizes
    sorted_df = filtered_df.sort_values(by='total_effic', ascending=False)
    traces = []
    annotations = []
    legend_names = {
        '0-9_yds': 'Short Pass (0-9 Yards) Efficiency',
        '10-19_yds': 'Medium Pass (10-19 Yards) Efficiency',
        '20+_yds': 'Deep Pass (20+ Yards) Efficiency'
    }
    for range in ['0-9_yds', '10-19_yds', '20+_yds']:
        sizes = filtered_df[f'{range}_Att_norm']
        traces.append(go.Scatter(
            x=filtered_df[f'{range}_Effic'],
            y=filtered_df['total_effic'],
            mode='markers',
            marker=dict(
                size= sizes*.6,
                opacity=0.6,
                line=dict(width=1, color='black')
            ),
            name=legend_names[range],
            text=filtered_df['Team']
        ))
    
    for i, team in enumerate(sorted_df['Team'].unique()):
        team_total_effic = sorted_df[sorted_df['Team'] == team]['total_effic'].values[0]
        traces.append(go.Scatter(
            x=[10, filtered_df['20+_yds_Effic'].max()+10],
            y=[team_total_effic, team_total_effic],
            mode='lines',
            line=dict(color='grey', dash='dot'),
            showlegend=False
        ))

        # Alternate annotation positions
        x_position = filtered_df['20+_yds_Effic'].max()+10 if i % 2 == 0 else 8
        annotations.append(dict(
            x=x_position,
            y=team_total_effic,
            xanchor='left' if i % 2 == 0 else 'right',
            text=team,
            showarrow=False
        ))

    return {
        'data': traces,
        'layout': go.Layout(
            title='Passing Efficiency by Team and by Depth',
            title_x=0.5,
            xaxis=dict(title='Efficiency by Depth'),
            yaxis=dict(title='Overall Efficiency'),
            hovermode='closest',
            colorway=px.colors.qualitative.Plotly,
            annotations=annotations
        )
    }

app.layout = html.Div([
    
#     style={'font-family': 'Arial, sans-serif', 'background-color': '#f4f4f2', 'padding': '20px'},
    # Title
    html.H1("CFL 2022-2023 Season Analysis", style={'text-align': 'center', 'color': '#2a3f5f', 'padding': '10px', 'background-color': 'white'}),

    # Scatter Plot Section
    html.Div([
        dcc.Slider(
            id='week-slider-scatter',
            min=1,
            max=21,
            step=1,
            value=1,
            marks={i: f'Week {i}' for i in range(1, 22, 2)}
        ),
        dcc.RadioItems(options=['Points', 'Yards'],
                       value='Yards',
                       inline=True,
                       id='radio-scatter',
                       style={'padding': '5px', 'margin': '10px'}
        ),
        dcc.Graph(id='scatter-plot'),
        html.P("The scatter plot above shows points for and points against, as well as yards for and yards against for each team throughout the seaason. As the season goes in, we can see teams move in directions we would generally expect. On the points comparison chart, the two strongest teams, Toronto and Winnipeg, were firmly in the bottom right corner, both far ahead of the pack in terms of Point Differential. Similarly, the weakest teams, Ottawa, Edmonton and Sasketchewan were firmly in the upper right corner, indicating weak offensive and defensive performance. When looking at the yardage chart, we see generally similar placements. Noticeably, Toronto find themselves much closer to the centre in both offense and defence, suggesting their strong point differential might have been driven by some scoring luck or other factors, since they did not outgain their opponents to a similar level as Winnipeg", style={'margin-top': '10px'})
    ], style={'border': '1px solid #ddd', 'box-shadow': '2px 2px 2px lightgrey', 'padding': '20px', 'margin-bottom': '30px', 'background-color': 'white'}),
    
    # Rushing-Passing Section
    html.Div([
        dcc.RadioItems(
            id='offense-defense-selector-rushing-passing',
            options=[
                {'label': 'Offense', 'value': 'Offense'},
                {'label': 'Defense', 'value': 'Defense'}
            ],
            value='Offense'
        ),
        dcc.Dropdown(
            id='rushing-passing-selector',
            options=[
                {'label': 'Rushing', 'value': 'Rushing'},
                {'label': 'Passing', 'value': 'Passing'}
            ],
            value='Rushing'
        ),
        dcc.Graph(
            id='rushing-passing',
            style={'max-width': '50%'}
        ),
        html.P("The visualization above shows how the season progressed for each team's rushing and passing offense and defense. Again, we can see Winnipeg was head and shoulders above the rest of the league, being top 3 in every category. Edmonton and Ottawa were both especially week in the passing game, being towards the bottom on both offense and defense. BC's offense was heavily polarized, placing first in passying yards for, but last in rushing yards for. ", style={'margin-top': '10px'})
    ], style={'border': '1px solid #ddd', 'box-shadow': '2px 2px 2px lightgrey', 'padding': '20px', 'margin-bottom': '30px', 'background-color': 'white'}),
    
    # First Down Section
    html.Div([
        dcc.Dropdown(
            id='week-dropdown-first-down',
            options=[{'label': f'Week {i}', 'value': i} for i in range(1, 22)],
            value=1
        ),
        dcc.Graph(id='first-down'),
        html.P("The graph above shows average yards gained on 1st down for each team, and the percentage they ran or threw the ball on first down. The CFL, unlike the NFL, is only a 3 down league, so every yard gained on first down is incredibly important. As expected, the teams with strong offenses would average more yards on first down. The three strongest first down teams, Toronto and Winnipeg, ran the ball a large chunk of the time (over 45%), suggesting a strong first down rush game led to strong first down success. However, the only other team to average over 7 yards, BC, only ran the ball on 32.3% of the time, but was still able to be succesful with a strong first down passing gaame.", style={'margin-top': '10px'})
    ], style={'border': '1px solid #ddd', 'box-shadow': '2px 2px 2px lightgrey', 'padding': '20px', 'margin-bottom': '30px', 'background-color': 'white'}),

    
    # Second Down Section
    html.Div([
        dcc.Slider(
            id='week-slider-second-down',
            min=1,
            max=21,
            step=1,
            value=1,
            marks={i: f'Week {i}' for i in range(1, 22, 2)}
        ),
        dcc.RadioItems(
            id='heatmap-type',
            options=[
                {'label': 'Offense', 'value': 'offense'},
                {'label': 'Defense', 'value': 'defense'}
            ],
            value='offense'
        ),
        dcc.Graph(id='second-down'),
        html.P("As mentioned above, the CFL only plays with three downs. For that reason, second down conversions are incredibly important. Converting on second down allows you to keep your drive going, and allows you to avoid potential punts or turnovers-on-downs. The heatmap above shows how teams performed at converting on second down on offence, as well as how well their defence was at preventing teams from converting on second down. We can see the strong offenses, like Winnipeg and BC were strong second down teams. BC and Hamilton often faced long second downs, but were also top in the league in converting on second and log. On defense, simlarly the teams that performed well defensively overall had strong second down defence. Toronto and Winnpeg forced teams to face the longest second downs, and stopped them from converting fairly regularly. Some of the weaker teams, like Edmonton and Saskatchewan, fared particularly poorly on stopping second and short attempts. That, paired with poor first down defence, can explain their subpar defensive results. This may help explain why these two teams ranked bottom 2 in both points against and yards against.", style={'margin-top': '10px'})
    ], style={'border': '1px solid #ddd', 'box-shadow': '2px 2px 2px lightgrey', 'padding': '20px', 'margin-bottom': '30px', 'background-color': 'white'}),
    
    # Aggression Section
    html.Div([
        dcc.Dropdown(
            id='week-dropdown-agression',
            options=[{'label': f'Week {i}', 'value': i} for i in range(1, 22)],
            value=1
        ),
        dcc.Graph(id='agression'),
        html.P("Every fan loves watching aggresive football. So I decided to take a look at certain agressive plays, and graph them compared to a team's average points for per game. The plays selected were two-point conversion attempts, third and short conversion attempts, and deep pass attempts. These plays were chosen because they are all decisions that are generally considered agressive by football fans and analysts. The graph shows varying levels of connections. It seems that there wasn't a strong correlation between two-point conversion attempts and points scored, nor with third and short conversion attempts. However, we can clearly see that teams that attempted more deep passes did tend to average more points per game.", style={'margin-top': '10px'})
    ], style={'border': '1px solid #ddd', 'box-shadow': '2px 2px 2px lightgrey', 'padding': '20px', 'margin-bottom': '30px', 'background-color': 'white'}),

    # Kicking Section
    html.Div([
        dcc.Dropdown(
            id='team-dropdown-kicking',
            options=[{'label': f'Team: {Team}', 'value': Team} for Team in Teams],
            value='BC'
        ),
        dcc.RadioItems(
            id='kick-type',
            options=[
                {'label': 'Kickoffs', 'value': 'kickoffs'},
                {'label': 'Punts', 'value': 'punts'}
            ],
            value='punts'
        ),
        dcc.Graph(id='kicking'),
        html.P("Above we see a visualization of team's performance on kickoffs and punting. The above graph can help us see which teams were most effective at gaining yard position through strong special teams performace. Overall, there wasn't too much yardage to be gained on the punt or the kickoff. However some teams did consistently perform better or worse over the season. Montreal seemed to particularly excel at special teams. After kickoffs, opponents started around 36 yard on average, the second best mark in the leauge. They also allowed only one return for more than 40 yards over the all whole season. They were also very effective at punting, netting the most yards on average in the league.", style={'margin-top': '10px'})
    ], style={'border': '1px solid #ddd', 'box-shadow': '2px 2px 2px lightgrey', 'padding': '20px', 'margin-bottom': '30px', 'background-color': 'white'}),

    # Passing Range Section
    html.Div([
        dcc.RadioItems(
            id='offense-defense-selector-passing-range',
            options=[
                {'label': 'Offense', 'value': 'Offense'},
                {'label': 'Defense', 'value': 'Defense'},
            ],
            value = 'Offense'
        ),
        dcc.Graph(id='passing-range'),
        html.P("This chart shows how each team succeeded in passing efficiency, broken up by throwing range. Passing efficiency is a stat created to try and capture a team's overall passing performance. The stat takes into account attempts, completions, yards, TDs and interceptions. On the y-axis, each team is organized by their overall passing efficiency. Each team has a bubble showing their short, medium and deep range passing efficiency (represented on the x-axis). The size of the bubbles demonstrates the number of attempts each team threw at each range. The bubble sizes are normalized, so the size of the bubble shows how much teams throw at that depth relative to other teams at the same depth. There's a lot of interesting information on this chart, but we can clearly see that teams generally threw at similar levels of efficiency on short throws. The gains or losses in efficiency can be seen in how well and how often teams threw the ball medium and especially deep. Teams that had stronger overall passing efficiency tended to throw the ball deep more often, and more effictively. Similarly, on defense the teams that caused opponents to have the lowest overall efficiency were the ones who were best at keeping their opponents efficiency on deep passes limited. ", style={'margin-top': '10px'})
    ], style={'border': '1px solid #ddd', 'box-shadow': '2px 2px 2px lightgrey', 'padding': '20px', 'margin-bottom': '30px', 'background-color': 'white'}),
    html.Div([
        html.P("Developed by Jared Boretsky", style={'text-align': 'center', 'color': '#2a3f5f'})
    ], style={'padding': '10px', 'background-color': '#e8eaf6'})
],style={'font-family': 'Raleway:Lato', 'color': '#003366', 'background-color': '#8FA08A', 'padding': '20px'})

if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




