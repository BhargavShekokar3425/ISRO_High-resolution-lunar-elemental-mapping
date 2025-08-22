import pandas as pd
import plotly.graph_objects as go
from scipy.interpolate import griddata
import numpy as np
from dash import Dash, html, dcc, Output, Input, callback_context
import os
import mysql.connector
from dash.dependencies import Output, Input, State
from PIL import Image

# File path for the dataset
# FILE_PATH = 'kriged_results_kriging_ratios_1.csv'  # Adjust the path accordingly

def get_db_connection():
    connection = mysql.connector.connect(
        host='localhost',       # Adjust as needed
        user='root',
        password='1234',
        database='isro_team_73',
        port = 3306
    )
    return connection

def get_db_connection2():
    connection = mysql.connector.connect(
        host='localhost',       # Adjust as needed
        user='root',
        password='1234',
        database='isro_team_73',
        port = 3306
    )
    cur = connection.cursor()
    return (connection, cur)

data_cache = pd.DataFrame()

def load_all_data():
    global data_cache
    connection = get_db_connection()
    query = "SELECT * FROM kriged_data"
    data_cache = pd.read_sql(query, connection)
    connection.close()
    data_cache.rename(columns = {"mg/si_kriged":"mg_si", "al/si_kriged":"al_si", "ca/si_kriged":"ca_si"}, inplace = True)

def load_updated_data():
    connection, cur = get_db_connection()
    query = "SELECT * FROM update_data"
    updated_data = pd.read_sql(query, connection)
    query = "DELETE * FROM update_data"
    cur.execute(query)
    updated_data.rename(columns = {"mg/si_kriged":"mg_si", "al/si_kriged":"al_si", "ca/si_kriged":"ca_si"}, inplace = True)
    connection.commit()

    cur.close()
    connection.close()
    return updated_data

# def update_grid_with_changes(changed_rows, new_data, grid, lon_grid, lat_grid):
#     """Update only the affected grid points."""
#     updated_points = new_data.loc[changed_rows, ['centroid_lon', 'centroid_lat']].to_numpy()
#     updated_values = new_data.loc[changed_rows, 'normalized_value'].to_numpy()
    
#     # Interpolate updated points
#     updated_abundance_grid = griddata(
#         points=updated_points,
#         values=updated_values,
#         xi=(lon_grid, lat_grid),
#         method='linear',
#         fill_value=0
#     )
    
#     # Merge updated grid values into the existing grid
#     mask = updated_abundance_grid != 0  # Update only non-zero values
#     grid[mask] = updated_abundance_grid[mask]
#     return grid




def create_basemap_sphere():
    # Load the image
    basemap_image = Image.open('lroc_color_poles_1k.jpg').convert('RGB')
    img_array = np.array(basemap_image)
    img_height, img_width, _ = img_array.shape

    # Create lat/lon grid
    lat_grid = np.linspace(-90, 90, 180)
    lon_grid = np.linspace(-180, 180, 360)
    lon_grid_mesh, lat_grid_mesh = np.meshgrid(lon_grid, lat_grid)

    # Convert lat/lon to pixel coordinates
    # Longitude: -180 -> 0 pixel, +180 -> max pixel
    # Latitude: +90 -> 0 pixel (top), -90 -> bottom
    # row = 0 at top (lat=90), row = img_height-1 at bottom (lat=-90)
    # col = 0 at lon=-180, col=img_width-1 at lon=180
    row = ((90 - lat_grid_mesh) / 180) * (img_height - 1)
    col = ((lon_grid_mesh + 180) / 360) * (img_width - 1)

    # Round and convert to int
    row = np.round(row).astype(int)
    col = np.round(col).astype(int)

    # Ensure indices are in range
    row = np.clip(row, 0, img_height - 1)
    col = np.clip(col, 0, img_width - 1)

    # Extract RGB values and convert to grayscale
    # Simple luminosity: gray = 0.299*R + 0.587*G + 0.114*B
    R = img_array[row, col, 0].astype(float)
    G = img_array[row, col, 1].astype(float)
    B = img_array[row, col, 2].astype(float)
    gray = 0.299*R + 0.587*G + 0.114*B

    # Normalize gray to [0,1]
    gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-9)

    # Convert lat/lon to 3D sphere coordinates (radius slightly less than data sphere)
    radius = 0.99  # Slightly smaller so it's "under" the main surface
    lat_rad = np.radians(lat_grid_mesh)
    lon_rad = np.radians(lon_grid_mesh)
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    # Create the surface trace for the basemap
    basemap_trace = go.Surface(
        x=x,
        y=y,
        z=z,
        surfacecolor=gray,
        colorscale='Gray',  # or a custom colorscale
        cmin=0,
        cmax=1,
        showscale=False,
        hoverinfo='skip',  # No hover for basemap
        opacity=1
    )

    return basemap_trace

# Function to process data and generate the spherical heatmap

def process_3d_data(col):
    try:
        global data_cache

        data = data_cache.copy()
    
        # Convert necessary columns to numeric types
        cols_to_convert = [
            'u_l_lat', 'u_r_lat', 'b_r_lat', 'b_l_lat',
            'u_l_lon', 'u_r_lon', 'b_r_lon', 'b_l_lon',
            'mg_si', 'al_si', 'ca_si',
            'cent_lat', 'cent_lon'
        ]
        for col_name in cols_to_convert:
            data[col_name] = pd.to_numeric(data[col_name], errors='coerce')
    
        # Remove rows with missing data
        data = data.dropna(subset=['cent_lat', 'cent_lon', col])
    
        # Specify the column to process
        data['normalized_value'] = data[col]
    
        # Clip and normalize the data
        data['normalized_value'] = data['normalized_value'].clip(upper=2, lower=0)
        data['normalized_value'] = (data['normalized_value'] - data['normalized_value'].min()) / \
                                   (data['normalized_value'].max() - data['normalized_value'].min())
    
        # Create a grid in latitude and longitude
        lat_grid = np.linspace(-90, 90, 180)
        lon_grid = np.linspace(-180, 180, 360)
        lon_grid, lat_grid = np.meshgrid(lon_grid, lat_grid)
    
        # Interpolate data onto the grid
        points = np.array([data['cent_lon'], data['cent_lat']]).T
        values = data['normalized_value']
        grid_z = griddata(points, values, (lon_grid, lat_grid), method='linear')
        # Handle NaNs in grid_z
        grid_z = np.nan_to_num(grid_z, nan=0.0)
    
        # Convert grid to spherical coordinates
        lat_rad = np.radians(lat_grid)
        lon_rad = np.radians(lon_grid)
        radius = 1
        x = radius * np.cos(lat_rad) * np.cos(lon_rad)
        y = radius * np.cos(lat_rad) * np.sin(lon_rad)
        z = radius * np.sin(lat_rad)

        # Create the 3D spherical heatmap
        fig = go.Figure()

        # Calculate latitude and longitude from x, y, z for hovertemplate
        hover_lat = np.degrees(np.arcsin(z / radius))
        hover_lon = np.degrees(np.arctan2(y, x))
    
        # Construct the 'text' array using np.vectorize
        text = np.vectorize(lambda lat, lon, val: f"Latitude: {lat:.2f}°<br>Longitude: {lon:.2f}°<br>Value: {val:.2f}")(
            hover_lat, hover_lon, grid_z
        )
        basemap_trace = create_basemap_sphere()
        
        fig.add_trace(basemap_trace)

        fig.add_trace(
            go.Surface(
                x=x,
                y=y,
                z=z,
                surfacecolor=grid_z,
                colorscale='Jet',
                cmin=0,
                cmax=1.0,
                colorbar=dict(title="Normalized Abundance"),
                hoverinfo='text',
                opacity=0.4,
                text=text
            )
        )
    
        # Update layout for visualization
        fig.update_layout(
            title=f"3D Heatmap of {col}",
            scene=dict(
                xaxis=dict(visible=False),
                yaxis=dict(visible=False),
                zaxis=dict(visible=False),
                aspectmode='data',
                aspectratio=dict(x=1, y=1, z=1),
                # camera_eye=dict(x=1.5, y=1.5, z=1.5),
                dragmode='orbit',
            ),
            paper_bgcolor='black',
            font=dict(color='white', family='Orbitron, sans-serif', size=18),
            margin=dict(l=0, r=0, t=50, b=0),
        )

        # Add rotation animation
        frames = []
        for angle in range(0, 360, 5):
            frames.append(go.Frame(layout=dict(
                scene_camera_eye=dict(
                    x=1.5 * np.cos(np.radians(angle)),
                    y=1.5 * np.sin(np.radians(angle)),
                    z=1.5
                )
            )))

        fig.frames = frames

        fig.update_layout(
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    buttons=[
                        dict(
                            label="Auto Rotate",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 50, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                    "mode": "immediate"
                                }
                            ],
                        )
                    ],
                    x=0.1,
                    y=0.1,
                    bgcolor='black',
                    bordercolor='white',
                    font=dict(color='white')
                )
            ]
        )

        return fig

    except Exception as e:
        print(f"Error processing data: {e}")
        return go.Figure()
    

def process_2d_data(col):
    try:
        global data_cache

        data = data_cache.copy()
        data['normalized_value'] = data[col]

        # Normalize values
        data['normalized_value'] = data['normalized_value'].clip(upper=2, lower=0)
        if data['normalized_value'].max() - data['normalized_value'].min() != 0:
            data['normalized_value'] = (data['normalized_value'] - data['normalized_value'].min()) / \
                                    (data['normalized_value'].max() - data['normalized_value'].min())
        else:
            data['normalized_value'] = 0

        # Create figure
        fig = go.Figure()

        # Add basemap image (assuming you have a basemap image file)
        # You need to adjust the path to your basemap image
        basemap_image_path = 'lroc_color_poles_1k.jpg'  # Replace with your basemap image path
        basemap_image = Image.open(basemap_image_path)

        fig.add_layout_image(
            dict(
                source=basemap_image,
                xref="x",
                yref="y",
                x=-180,    # Left longitude
                y=90,      # Top latitude
                sizex=360, # Width (longitude range)
                sizey=180, # Height (latitude range)
                sizing="stretch",
                opacity=1,
                layer="below"
            )
        )

        fig.add_trace(go.Heatmap(
            x=data['cent_lon'],  # Longitudes
            y=data['cent_lat'],  # Latitudes
            z=data['normalized_value'],  # Values to color-code
            colorscale='Jet',  # Color scale
            colorbar=dict(title="Normalized Value"),  # Color bar settings
            zmin=0,  # Minimum value for color scale
            zmax=1,   # Maximum value for color scale
            opacity=0.55
        ))

        # Adding hover text (optional)
        fig.update_traces(
            hovertemplate="Lat: %{y:.2f}<br>Lon: %{x:.2f}<br>Value: %{z:.2f}<extra></extra>"
        )

        # Configure layout
        fig.update_layout(
            title=f"2D Heatmap of {col}",
            xaxis=dict(
                title="Longitude",
                range=[-180, 180],
                showgrid=False,
                zeroline=False
            ),
            yaxis=dict(
                title="Latitude",
                range=[-90, 90],
                showgrid=False,
                zeroline=False
            ),
            paper_bgcolor='black',
            plot_bgcolor='black',
            font=dict(color='white', family='Orbitron, sans-serif', size=18),
            margin=dict(l=0, r=0, t=50, b=0),
            clickmode='event+select',
        )

        return fig

    except Exception as e:
        print(f"Error processing file: {e}")
        return go.Figure()
    
# Function to update the figure with changes (3D plot)
def update_3d_surface_plot(fig, updated_data, col):
    global data_cache

    # Update data_cache with updated_data
    data_cache.set_index('id', inplace=True)
    updated_data.set_index('id', inplace=True)

    data_cache.update(updated_data)
    data_cache = data_cache.combine_first(updated_data)

    data_cache.reset_index(inplace=True)
    updated_data.reset_index(inplace=True)

    data = data_cache.copy()
    data['normalized_value'] = data[col]

    # Remove rows with missing data
    data = data.dropna(subset=['cent_lat', 'cent_lon', col])

    # Clip and normalize the data
    data['normalized_value'] = data['normalized_value'].clip(upper=2, lower=0)
    if data['normalized_value'].max() - data['normalized_value'].min() != 0:
        data['normalized_value'] = (data['normalized_value'] - data['normalized_value'].min()) / \
                                   (data['normalized_value'].max() - data['normalized_value'].min())
    else:
        data['normalized_value'] = 0

    # Create a grid in latitude and longitude
    lat_grid = np.linspace(-90, 90, 180)
    lon_grid = np.linspace(-180, 180, 360)
    lon_grid_mesh, lat_grid_mesh = np.meshgrid(lon_grid, lat_grid)

    # Interpolate data onto the grid
    points = np.array([data['cent_lon'], data['cent_lat']]).T
    values = data['normalized_value']
    grid_z = griddata(points, values, (lon_grid_mesh, lat_grid_mesh), method='linear', fill_value=0)

    # Convert grid to spherical coordinates
    lat_rad = np.radians(lat_grid_mesh)
    lon_rad = np.radians(lon_grid_mesh)
    radius = 1
    x = radius * np.cos(lat_rad) * np.cos(lon_rad)
    y = radius * np.cos(lat_rad) * np.sin(lon_rad)
    z = radius * np.sin(lat_rad)

    # Calculate latitude and longitude from x, y, z for hovertemplate
    hover_lat = np.degrees(np.arcsin(z / radius))
    hover_lon = np.degrees(np.arctan2(y, x))

    # Construct the 'text' array using np.vectorize
    text = np.vectorize(lambda lat, lon, val: f"Latitude: {lat:.2f}°<br>Longitude: {lon:.2f}°<br>Value: {val:.2f}")(
        hover_lat, hover_lon, grid_z
    )

    fig.add_trace(
        text=text
    )

    # Update the Surface plot data
    fig.data[1].x = x
    fig.data[1].y = y
    fig.data[1].z = z
    fig.data[1].surfacecolor = grid_z

    # Regenerate frames for rotation
    # frames = []
    # for angle in range(0, 360, 5):
    #     camera_eye = dict(
    #         x=1.5 * np.cos(np.radians(angle)),
    #         y=1.5 * np.sin(np.radians(angle)),
    #         z=1.25
    #     )
    #     frames.append(go.Frame(layout=dict(scene_camera_eye=camera_eye)))
    # fig.frames = frames

    print("Updating 3D surface plot...")

    return fig

def update_2d_heatmap_plot(fig, updated_data, col):
    global data_cache

    # Update data_cache with updated_data
    data_cache.set_index('id', inplace=True)
    updated_data.set_index('id', inplace=True)
    
    data_cache.update(updated_data)
    data_cache = data_cache.combine_first(updated_data)

    data_cache.reset_index(inplace=True)
    updated_data.reset_index(inplace=True)

    data = data_cache.copy()
    data['normalized_value'] = data[col]

    # Remove rows with missing data
    data = data.dropna(subset=['cent_lat', 'cent_lon', col])

    # Clip and normalize the data
    data['normalized_value'] = data['normalized_value'].clip(upper=2, lower=0)
    value_range = data['normalized_value'].max() - data['normalized_value'].min()
    if value_range != 0:
        data['normalized_value'] = (
            data['normalized_value'] - data['normalized_value'].min()
        ) / value_range
    else:
        data['normalized_value'] = 0

    # Update the heatmap data
    fig.data[0].x = data['cent_lon']
    fig.data[0].y = data['cent_lat']
    fig.data[0].z = data['normalized_value']


    print("Updating 2D heatmap plot...")
    # print(f"Updated data shape: {updated_data.shape}")

    return fig

# Main function to set up the Dash app
if __name__ == "__main__":
    # Create the Dash app
    app = Dash(__name__)

    load_all_data()

    # Define the layout
    app.layout = html.Div(children=[
        html.H1('Heatmap Visualization', style={'textAlign': 'center', 'color': 'white'}),
        html.Div([
            html.Button("Mg/Si 3D", id='mg_si_3d', n_clicks=0, className='dash-button'),
            html.Button("Al/Si 3D", id='al_si_3d', n_clicks=0, className='dash-button'),
            html.Button("Ca/Si 3D", id='ca_si_3d', n_clicks=0, className='dash-button'),
            html.Button("Fe_K/Si 3D", id='fe_k_si_3d', n_clicks=0, className='dash-button'),
            html.Button("Mg/Si 2D", id='mg_si_2d', n_clicks=0, className='dash-button'),
            html.Button("Al/Si 2D", id='al_si_2d', n_clicks=0, className='dash-button'),
            html.Button("Ca/Si 2D", id='ca_si_2d', n_clicks=0, className='dash-button'),
        ], style={'textAlign': 'center'}),
        html.Div(
            dcc.Graph(
                id='heatmap-graph',
                figure=process_3d_data('mg_si'),
                config={'displayModeBar': False, 'scrollZoom': True}
            ),
            className='graph-container'
        ),
        dcc.Interval(
            id='interval-component',
            interval=10*1000,  # Update every 10 seconds
            n_intervals=0
        ),
        dcc.Store(id='current-plot-info', data={'plot_type': '3D', 'data_key': 'mg_si'})
    ], style={'backgroundColor': 'black'})

    # Set up the callback to update the graph
    @app.callback(
        [Output('heatmap-graph', 'figure'),
         Output('current-plot-info', 'data')],
        [Input('mg_si_3d', 'n_clicks'),
         Input('al_si_3d', 'n_clicks'),
         Input('ca_si_3d', 'n_clicks'),
         Input('fe_k_si_3d', 'n_clicks'),
         Input('mg_si_2d', 'n_clicks'),
         Input('al_si_2d', 'n_clicks'),
         Input('ca_si_2d', 'n_clicks'),
         Input('interval-component', 'n_intervals')],
        [State('heatmap-graph', 'figure'),
         State('current-plot-info', 'data')]
    )
    def update_graph(mg_si_3d_clicks, al_si_3d_clicks, ca_si_3d_clicks, fe_k_si_3d_clicks,
                     mg_si_2d_clicks, al_si_2d_clicks, ca_si_2d_clicks,
                     n_intervals, current_fig, current_plot_info):
        # print(f"Callback triggered. Figure: {current_fig}, Plot info: {current_plot_info}")
        ctx = callback_context
        triggers = ctx.triggered

        # Map buttons to data columns and processing functions
        button_ids = ['mg_si_3d', 'al_si_3d', 'ca_si_3d', 'fe_k_si_3d', 'mg_si_2d', 'al_si_2d', 'ca_si_2d']
        col_map = {
            'mg_si_3d': ('mg_si', process_3d_data, update_3d_surface_plot, '3D'),
            'al_si_3d': ('al_si', process_3d_data, update_3d_surface_plot, '3D'),
            'ca_si_3d': ('ca_si', process_3d_data, update_3d_surface_plot, '3D'),
            'fe_k_si_3d': ('fe_k_si', process_3d_data, update_3d_surface_plot, '3D'),
            'mg_si_2d': ('mg_si', process_2d_data, update_2d_heatmap_plot, '2D'),
            'al_si_2d': ('al_si', process_2d_data, update_2d_heatmap_plot, '2D'),
            'ca_si_2d': ('ca_si', process_2d_data, update_2d_heatmap_plot, '2D'),
        }

        if triggers:
            triggered_prop_ids = [trigger['prop_id'] for trigger in triggers]
            # Check if any button id is in triggered_prop_ids
            button_pressed = None
            for prop_id in triggered_prop_ids:
                button_id = prop_id.split('.')[0]
                if button_id in button_ids:
                    button_pressed = button_id
                    break

            if button_pressed:
                # A button was pressed; prioritize button actions
                data_key, processing_function, update_function, plot_type = col_map[button_pressed]
                current_plot_info = {'plot_type': plot_type, 'data_key': data_key}

                # Load all data and process the figure
                load_all_data()
                fig = processing_function(data_key)
                return fig, current_plot_info
        
        # If no button was pressed or no triggers, proceed with interval update
        if current_plot_info is None:
            # Default to initial plot info
            current_plot_info = {'plot_type': '3D', 'data_key': 'mg_si'}

        updated_data = load_updated_data()
        if not updated_data.empty:
            # Update the figure based on the current plot info
            fig = go.Figure(current_fig)
            plot_type = current_plot_info['plot_type']
            data_key = current_plot_info['data_key']

            # Choose the correct update function based on plot type
            if plot_type == '3D':
                update_function = update_3d_surface_plot
            else:
                update_function = update_2d_heatmap_plot

            fig = update_function(fig, updated_data, data_key)
            # Reset updated flags
            return fig, current_plot_info
        else:
            # No updates; return the current figure and plot info
            return current_fig, current_plot_info

    # Run the app on localhost
    app.run_server(debug=True)