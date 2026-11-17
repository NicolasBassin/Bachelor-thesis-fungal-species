import networkx as nx
import polars as pl
import pandas as pd
import fastexcel
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
import fungi_speci.utils_image as uim
import re
import cv2
import skimage as ski
from scipy.spatial import ConvexHull
import alphashape
from shapely.geometry import LineString, Point, box
from shapely.ops import unary_union


# from fungi_speci.branch_detection import branch_detection_by_max_width
from fungi_speci.load_data import get_network_from_static_final

class MyceliumGraph:

    def __init__(self, *args) -> None:
        self.set_graph_nodes_edges(args)
        self.add_degree_to_node_info()
        # positions = {node_name: [x, y], ...} and width = [width, ...]
        self.pos = self.get_positions()
        self.width = width_from_graph(self.G)
        self.add_direction_edges()
        # bookkeeping
        self.id_to_name = {i: node_name for i, node_name in enumerate(self.G.nodes)}
        self.name_to_id = {name: id for id, name in self.id_to_name.items()}
        # add "id" column to nodes dataframe (using polars map_elements)
        self.nodes = self.nodes.with_columns( id = self.nodes['name'].map_elements(
            lambda x: self.name_to_id[x], return_dtype=pl.Int32))

    def set_graph_nodes_edges(self, args):
        if len(args) == 1:
            suffix = pathlib.Path(args[0]).suffix.lower()
            if suffix == '.xlsx':
              self.G, self.nodes, self.edges, self.params = get_graph_from_xlsx(args[0])
            elif suffix in {'.h5', '.hdf5'}:
                self.G, self.nodes, self.edges = get_graph_from_hdf5(args[0])
            else:
                assert False, f"unsupported file type: {suffix}"
        elif len(args) == 2:
            if 'endnodes_1' in args[0].columns:
                self.edges, self.nodes = args
            elif 'endnodes_1' in args[1].columns:
                self.nodes, self.edges = args
            else:
                assert False, "no edges dataframe found"
            self.G = get_graph_from_edges_nodes(self.edges, self.nodes)
        else:
            assert False, "wrong number of arguments (either (filename) or (nodes, edges))"

    def add_image(self, image, resize_factor=1, offset_x=0, offset_y=0):
        '''add an image to the graph
        '''
        self.image = uim.load_image_gray(image)
        # scale the graph to the image (values by trial and error for 'neurospora_1_vm4_fc_Merged-results')
        # resize_factor (=params.resample), offset_x, offset_y = 1.8, 365, 273
        if hasattr(self, 'params'):
            resize_factor = self.params['resample'][0]
        sign = -1 if resize_factor != 1 else 1
        self.pos_image = {k: [v[0]*resize_factor + offset_x,
                              sign * v[1]*resize_factor + offset_y]
                          for k, v in self.pos.items()}

    def img_estimate_block_size(self):
        pixel_per_micron = self.params['resample'][0] / self.params['micron_per_pixel'][0]
        block_size = int(3 * pixel_per_micron * np.percentile(self.edges['width'], 95)) | 1
        return block_size

    def img_binarize(self, block_size, offset=0, with_inoculum_mask=True):
        '''binarize the image and save it as self.image_binary
        parameters:
            block_size: size of the block for local thresholding
            offset: offset for local thresholding
            with_inoculum_mask: whether to use the inoculum mask
        Notes:
            * block_size: the block size should be significant larger
                than the largest width of a branch
                * rule of thumb: block_size = 3 * 95th-percentile(width)
            * creates:
                * image_binary: np.ndarray, dtype=bool, True where branches are
                * inoculum_mask: np.ndarray, dtype=bool, True where inoculum is
        '''
        self.image_binary = uim.binarize(self.image, block_size, offset=offset)
        if with_inoculum_mask:
            self.inoculum_mask = uim.get_inoculum_mask(self.image, block_size)
            self.image_binary[self.inoculum_mask.astype(bool)] = False
        self.image_binary = uim.keep_largest_blob(self.image_binary)
        # optionally the inoculum are can now be removed
        # (keeping was important for keep_largest_blob to work)
        self.image_binary[self.inoculum_mask.astype(bool)] = False

    def get_positions(self):
        positions = {}
        for  ii, x, y in self.nodes[['name', 'x', 'y']].iter_rows():
            positions[ii] = [x, y]
        return positions

    def get_subgraph_from_edges(self, edges):
        nodes_to_keep = np.unique(edges[['endnodes_1', 'endnodes_2']].to_numpy().flatten())
        nodes = self.nodes.filter(pl.col('name').is_in(nodes_to_keep))
        myceg = MyceliumGraph(edges, nodes)
        return myceg

    def get_subgraph_from_node_names(self, node_names):
        edges = self.edges.filter(
            pl.col('endnodes_1').is_in(node_names) |
            pl.col('endnodes_2').is_in(node_names))
        return self.get_subgraph_from_edges(edges)

    def get_subgraph_from_edge_names(self, edge_names):
        edges = self.edges.filter(pl.col('name').is_in(edge_names))
        return self.get_subgraph_from_edges(edges)

    def get_subgraph_with_largest_edges(self, fraction_of_edges=0.1):
        '''return a subgraph with the largest edges
        '''
        N_edges = int(len(self.edges) * fraction_of_edges)
        edges = self.edges.sort('width').tail(N_edges)
        return self.get_subgraph_from_edges(edges)

    def get_subgraph_leafes(self):
        '''return a subgraph with the leafes
        '''
        leafes = [n for n, k in dict(self.G.degree()).items() if k == 1]
        edges = self.edges.filter(pl.col('endnodes_1').is_in(leafes) | pl.col('endnodes_2').is_in(leafes))
        return self.get_subgraph_from_edges(edges)

    def draw(self, fax=None, show_image=False, show_binary=False, width_scale=0.02, **kwargs):
        if fax is None:
            fax = plt.subplots(1)
        f, ax = fax
        plot_kwargs = dict(with_labels=False,
                           node_size=0.005,
                           node_color='w',
                           edge_color='gray')
        positions = self.pos
        if show_image:
            if not hasattr(self, 'image'):
                raise ValueError('image is None please run add_image() first')
            ax.imshow(self.image, cmap='gray', aspect='equal', alpha=0.5)
        if show_binary:
            if not hasattr(self, 'image_binary'):
                raise ValueError('image_binary is None please run img_binarize() first')
            ax.imshow(~self.image_binary, cmap='gray', aspect='equal', alpha=0.5)
        if show_image or show_binary:
            plot_kwargs['edge_color'] = 'r'
            plot_kwargs['node_color'] = 'r'
            positions = self.pos_image
        plot_kwargs.update(kwargs)
        nx.draw(self.G, positions, ax=ax, width=width_scale*self.width,
                **plot_kwargs)
        ax.set_aspect('equal')
        return f, ax

    def draw_image_with_inoculum_mask(self, fax=None):
        if fax is None:
            fax = plt.subplots(1)
        f, ax = fax
        if not hasattr(self, 'image'):
            raise ValueError('image is None please run add_image() first')
        ax.imshow(self.image, cmap='gray', aspect='equal')
        if not hasattr(self, 'inoculum_mask'):
            raise ValueError('inoculum_mask is None please run img_binarize() first')
        ax.imshow(self.inoculum_mask, cmap='Reds', aspect='equal', alpha=0.2)
        # show the contour of the inoculum mask
        contours = ski.measure.find_contours(self.inoculum_mask.astype(float), 0.5)
        for contour in contours:
            ax.plot(contour[:, 1], contour[:, 0], linewidth=0.6, color='r')
        ax.set_aspect('equal')
        ax.axis('off')
        return f, ax

    def get_node_closest_to_image_location(self, image_location):
        if not hasattr(self, 'image'):
            raise ValueError('image is None please run add_image() first')
        if len(image_location) != 2:
            raise ValueError('image_location should be a list or array of length 2')
        pos_image_np = np.array(list(self.pos_image.values()))
        dist_2_location = np.linalg.norm(pos_image_np - np.array(image_location), axis=1)
        dic_idx = np.argmin(dist_2_location)
        node_name = list(self.pos_image.keys())[dic_idx]
        node_idx = np.argwhere(self.nodes["name"] == node_name)[0][0]
        return node_name, node_idx

    def draw_node(self, node_name, fax=None, show_image=True,
                  size=700, node_size=1, **kwargs):
        """ draws the node and an area around it with the image in the background
        """
        f, ax = self.draw(fax=fax, show_image=show_image,
                          node_size=node_size, **kwargs)

        positions = self.pos
        if show_image and hasattr(self, 'image'):
            positions = self.pos_image
        center = positions[node_name]
        if show_image:
            if self.image is None:
                raise ValueError('image is None please run add_image() first')
            center = positions[node_name]
        ax.set(xlim=(center[0]-size/2, center[0]+size/2),
               ylim=(center[1]+size/2, center[1]-size/2))
        # make scatter but with empty circles
        ax.scatter(center[0], center[1], s=15, color='k', alpha=0.5)
        # now get the edges that are connected to the node
        edges = self.edges.filter(pl.col('endnodes_1') == node_name)
        edges = pl.concat([edges, self.edges.filter(pl.col('endnodes_2') == node_name)])
        # draw rectangles for each edge with the width
        for i, (name, n1, n2, width, phi) in enumerate(edges[['name', 'endnodes_1', 'endnodes_2', 'width', 'phi']].iter_rows()):
            if n1 == node_name:
                x1, y1 = positions[n1]
                x2, y2 = positions[n2]
            else:
                x2, y2 = positions[n1]
                x1, y1 = positions[n2]
                phi += np.pi
            phi *= 180 / np.pi
            length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            # draw a rectangle
            for factor in [-1, 1]:
                rect = plt.Rectangle((x1, y1), length, factor * width/2,
                                     angle=np.arctan2(y2-y1, x2-x1)*180/np.pi,
                                     color='g', alpha=0.2)
                ax.add_patch(rect)
        return f, ax

    def add_length_edges(self):
        self.add_xy_edges()
        self.edges = self.edges.with_columns(
            length = ((pl.col('x1') - pl.col('x2'))**2 + (pl.col('y1') - pl.col('y2'))**2).sqrt()
            )
        self.rm_xy_edges()

    def add_direction_edges(self):
        self.add_xy_edges()
        self.edges = self.edges\
            .with_columns(phi = np.arctan2(pl.col('y2') - pl.col('y1'), pl.col('x2') - pl.col('x1')))
        self.rm_xy_edges()

    def add_xy_edges(self):
        if 'x1' in self.edges.columns:
            return
        for i in [1, 2]:
            self.edges = self.edges\
                .join(self.nodes['name', 'x', 'y'], how='left', left_on=f'endnodes_{i}', right_on='name')\
                .rename({'x': f'x{i}', 'y': f'y{i}'})\

    def rm_xy_edges(self):
        if 'x1' not in self.edges.columns:
            return
        self.edges = self.edges.drop(['x1', 'y1', 'x2', 'y2'])

    # def branch_detection(self, strategy='max_width', **kwgs):
    #     existing_strategies = ['max_width']
    #     assert strategy in existing_strategies, f"strategy {strategy} not in {existing_strategies}"
    #     if strategy == 'max_width':
    #         branches = branch_detection_by_max_width(self.edges, **kwgs)
    #     return branches

    def get_start_node_of_branch(self, branch):
        start_nodes_0 = self.edges.filter(pl.col('name') == branch[0])[['endnodes_1', 'endnodes_2']].row(0)
        start_nodes_1 = self.edges.filter(pl.col('name') == branch[1])[['endnodes_1', 'endnodes_2']].row(0)
        start_node = start_nodes_0[0] if start_nodes_0[1] in start_nodes_1 else start_nodes_0[1]
        return start_node

    def draw_branch_ends(self, branch, distance_threshold=1, show_image=False):
        f, axs = plt.subplots(1, 4, figsize=(20, 5), gridspec_kw={'width_ratios': [1, 1, 2, 2]})
        n_ = []

        if show_image:
            if self.image is None:
                raise ValueError('image is None please run add_image() first')
            ax.imshow(self.image, cmap='gray', aspect='equal', alpha=0.5)
        for i, sub_branch in enumerate([branch[:distance_threshold+1],
                                      branch[::-1][:distance_threshold+1]]):
            ax = axs[i]
            start_node = self.get_start_node_of_branch(sub_branch)
            nodes_to_include = [start_node] + list(self.get_nearest_neighbors(start_node, dist=distance_threshold))
            n_ += nodes_to_include
            myceg_other = self.get_subgraph_from_node_names(nodes_to_include)
            myceg_branch = self.get_subgraph_from_edge_names(sub_branch)
            myceg_other.draw(fax=[f, ax], width_scale=0.5, alpha=0.3, node_color='k')
            myceg_branch.draw(fax=[f, ax], width_scale=0.5, edge_color='red')
            x, y = myceg_branch.pos[start_node]
            ax.scatter(x, y, color='k', s=5, marker='^')
            ax.set_aspect('equal')
        for ax in axs[2:]:
            self.draw(fax=[f, ax], alpha=0.1)
            myceg_branch = self.get_subgraph_from_edge_names(branch)
            myceg_branch.draw(fax=[f, ax], edge_color='red')
            ax.set_aspect('equal')
        ax = axs[2]
        xlim, ylim = myceg_branch.get_limits(extension_factor=1, rectangle=True)
        lim, _ = self.get_limits(rectangle=True)
        s_extra = 0.1*(lim[1] - lim[0])
        xlim = [xlim[0] - s_extra, xlim[1] + s_extra]
        ylim = [ylim[0] - s_extra, ylim[1] + s_extra]
        ax.set(xlim=xlim, ylim=ylim)
        return f, axs

    def get_nearest_neighbors(self, i, dist=1):
        nn = []
        _get_nearest_neighbors_(self.G, i, nn, dist=dist)
        return np.unique(nn)

    def get_limits(self, extension_factor=1, rectangle=False):
        x, y = self.nodes[['x', 'y']].to_numpy().T
        x_cen = (x.max() + x.min()) / 2
        y_cen = (y.max() + y.min()) / 2
        d_x = (x.max() - x.min()) * extension_factor
        d_y = (y.max() - y.min()) * extension_factor
        d_max = max(d_x, d_y)
        if rectangle:
            d_x = d_y = d_max
        return [x_cen - d_x / 2, x_cen + d_x / 2], [y_cen - d_y / 2, y_cen + d_y / 2]

    def get_edges_too_short(self, threshold):
        edges_to_remove = self.edges.with_columns(length_fraction = pl.col('length') / pl.col('width'))\
            .filter(pl.col('length_fraction') <= threshold)\
            .sort('length_fraction')\
            .drop('length_fraction')
        return edges_to_remove



    def add_degree_to_node_info(self):
        '''add the degree of the node to the node info
        '''
        degrees = dict(self.G.degree())
        self.nodes = self.nodes.with_columns(
            degree = pl.col('name').replace_strict(
                degrees, return_dtype=pl.Int32))
        return self.nodes



def _get_nearest_neighbors_(G, i, nn, dist=1):
    neighbors = list(G.neighbors(i))
    new_nn = [n for n in neighbors if n not in nn]
    nn += new_nn
    dist = dist - 1
    if dist > 0:
        for n in new_nn:
            _get_nearest_neighbors_(G, n, nn, dist=dist)


def get_graph_from_edges_nodes(edges, nodes):
    edge_attr_cols = [col for col in edges.columns if col not in {'endnodes_1', 'endnodes_2'}]
    edges_dict = {
        (e1, e2): {col: value for col, value in zip(edge_attr_cols, values)}
        for e1, e2, *values in edges[['endnodes_1', 'endnodes_2'] + edge_attr_cols].iter_rows()
    }
    node_attr_cols = [col for col in nodes.columns if col != 'name']
    nodes_dict = {
        id: {col: value for col, value in zip(node_attr_cols, values)}
        for id, *values in nodes[['name'] + node_attr_cols].iter_rows()
    }
    G = nx.from_edgelist(edges_dict, create_using=nx.Graph)
    nx.set_edge_attributes(G, edges_dict)
    nx.set_node_attributes(G, nodes_dict)
    return G


def get_graph_from_xlsx(filename):
    '''get from an xlsx file the sheet "edges" and "nodes"
        and return the graph
    '''
    workbook = fastexcel.read_excel(filename)
    # Construct DataFrames explicitly.  ``pl.read_excel`` currently calls
    # ``pl.from_arrow`` for this engine, which warns that its return type will
    # change in Polars 2.0 for ArrowStreamExportable inputs.
    edges = pl.DataFrame(workbook.load_sheet_by_name("Edges"))
    nodes = pl.DataFrame(workbook.load_sheet_by_name("Nodes"))
    params = pl.DataFrame(workbook.load_sheet_by_name("Parameters"))
    # load with polars the excel files
    nodes = nodes[['node_ID', 'node_X_pix', 'node_Y_pix', 'node_Type']]\
                 .rename({'node_ID': 'name', 'node_X_pix': 'x', 'node_Y_pix': 'y'})\
                 .with_columns(y = -pl.col('y'))
    edges = edges[['EndNodes_1', 'EndNodes_2', 'Width', 'Length', 'Type', 'Name']]
    nodes = df_column_snake_name(nodes)
    edges = df_column_snake_name(edges)
    G = get_graph_from_edges_nodes(edges, nodes)
    return G, nodes, edges, params


def get_graph_from_hdf5(filename):
    edge_list, edge_info, node_info = get_network_from_static_final(filename)
    node_info = np.asarray(node_info, dtype=float)

    node_ids = np.arange(len(node_info), dtype=int)
    nodes = pl.DataFrame({
        'name': node_ids,
        'x': node_info[:, 0],
        'y': node_info[:, 1],
        'node_type': ['node'] * len(node_ids),
    })

    edge_array = np.asarray(edge_list, dtype=int)
    if edge_array.size == 0:
        edges = pl.DataFrame({
            'endnodes_1': pl.Series([], dtype=pl.Int64),
            'endnodes_2': pl.Series([], dtype=pl.Int64),
            'width': pl.Series([], dtype=pl.Float64),
            'length': pl.Series([], dtype=pl.Float64),
            'type': pl.Series([], dtype=pl.String),
            'name': pl.Series([], dtype=pl.Int64),
            'branch_id': pl.Series([], dtype=pl.Int64),
        })
    else:
        length = np.linalg.norm(
            node_info[edge_array[:, 1]] - node_info[edge_array[:, 0]],
            axis=1,
        )
        edges = pl.DataFrame({
            'endnodes_1': edge_array[:, 0],
            'endnodes_2': edge_array[:, 1],
            'width': np.ones(len(edge_array), dtype=float),
            'length': length,
            'type': ['simulation'] * len(edge_array),
            'name': np.arange(len(edge_array), dtype=int),
            'branch_id': np.asarray(edge_info, dtype=int),
        })

    G = get_graph_from_edges_nodes(edges, nodes)
    return G, nodes, edges


def width_from_graph(G):
    width = np.array([d['width'] for u, v, d in G.edges(data=True)])
    return width


def squezze_repeated_char(text, char):
    out = re.sub(f"({char})+", char, text)
    return out


def df_column_snake_name(df, sep="_"):
    polars = type(df) is pl.DataFrame
    # rename columns
    renamer = {
        c: c.replace(".", sep)
        .replace("…", sep)
        .replace("(", sep)
        .replace(")", "")
        .replace(" ", sep)
        .replace("%", f"in{sep}percent")
        .lower()
        .strip(sep)
        for c in df.columns
    }
    renamer = {k: squezze_repeated_char(v, sep) for k, v in renamer.items()}
    if polars:
        df = df.rename(renamer)
    else:
        df = df.rename(columns=renamer)
    return df

def hello_graph():
    print('Hi, I am a function of mynan/graph.py')

def convert_df_generated(file_hdf5):
    myceg_simu = MyceliumGraph(file_hdf5)
    df = pd.DataFrame(myceg_simu.nodes, columns=myceg_simu.nodes.columns)
    df['degree'] = df['degree'].astype(int)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['id'] = df['id'].astype(int)
    return df, myceg_simu

def convert_df_original(file_xlsx):
    myceg = MyceliumGraph(file_xlsx)
    df = pd.DataFrame(myceg.nodes, columns=myceg.nodes.columns)
    df['degree'] = df['degree'].astype(int)
    df['x'] = df['x'].astype(float)
    df['y'] = df['y'].astype(float)
    df['id'] = df['id'].astype(int)
    return df, myceg

def get_radial_dist(df, bool_choice):
    is_quarter_circle = bool_choice
    if is_quarter_circle:
        x_normal = df['x'] - df['x'].min()
        y_normal = df['y'] - df['y'].min()
        df['radial_distance'] = np.sqrt(x_normal**2 + y_normal**2)
    else:
        df['radial_distance'] = np.sqrt( df['x']**2 + df['y']**2)

    return df

def get_bin_borders_count(df, desired_bin_size, bool_choice):
    df= get_radial_dist(df, bool_choice)
    radius_network = df['radial_distance'].max() - df['radial_distance'].min()
    bin_number_circle = radius_network / desired_bin_size
    bin_number_circle = bin_number_circle.astype(int)
    count, bin_borders = np.histogram(df['radial_distance'], bins=bin_number_circle)
    bin_middle = (bin_borders[:-1] + bin_borders[1:]) / 2

    return count, bin_borders, bin_middle

def get_area_of_segments(df, desired_bin_size, bool_choice):
    is_quarter_circle = bool_choice
    count, bin_borders, bin_middle = get_bin_borders_count(df, desired_bin_size, bool_choice)
    full_areas = []
    for i in bin_borders:
        full_areas.append(np.pi * i**2)
    true_areas = []
    for i in range (1, len(full_areas)):
        true_areas.append(full_areas[i] - full_areas[i - 1])
    true_areas = np.array(true_areas)    
    if is_quarter_circle:
        true_areas /= 4
    return true_areas

def get_density(df, desired_bin_size, bool_choice):
    true_areas = get_area_of_segments(df, desired_bin_size, bool_choice)
    count, bin_borders, bin_middle = get_bin_borders_count(df, desired_bin_size, bool_choice)
    density = count / true_areas
    df_density = pd.DataFrame({
        'bin_middle' : bin_middle,
        'density' : density
    })
    return df_density

def get_angles_of_nodes(df, bool_choice):
    is_quarter_circle = bool_choice
    if is_quarter_circle:
        df['degree_to_center'] = abs(np.arctan2(df['y'] - df['y'].min(), df['x'] - df['x'].min()) * 180 /np.pi)
    else:
        df['degree_to_center'] = np.arctan2(df['y'], df['x']) * 180 /np.pi
    return df

def get_standard_deviation_of_density(desired_bin_size, desired_bin_size_angular, df, bool_choice):
    vgdf = get_angles_of_nodes(df, bool_choice)
    count, bin_borders, bin_middle = get_bin_borders_count(df, desired_bin_size, bool_choice)
    true_areas = get_area_of_segments(df, desired_bin_size, bool_choice)
    df_bins = create_df_bins(df, desired_bin_size, bool_choice)
    if bool_choice == True:
        bin_number_angle = 90 / desired_bin_size_angular
        bin_number_angle = int(bin_number_angle)
    else:
        bin_number_angle = 360 / desired_bin_size_angular
        bin_number_angle = int(bin_number_angle)
    stds_by_circle_segment = []

    for n in range(1, len(bin_borders)):
        seg_min = bin_borders[n - 1]
        seg_max = bin_borders[n]

        df_circle_segment = df.query(f'radial_distance >= {seg_min} and radial_distance < {seg_max}')

        if len(df_circle_segment) == 0:
            stds_by_circle_segment.append(np.nan)
            continue

        count_segment, _ = np.histogram(df_circle_segment['degree_to_center'], bins=bin_number_angle)
        area_segment = true_areas[n - 1] / bin_number_angle
        densities_segments = count_segment / area_segment
        std_density = np.std(densities_segments, ddof=1)
        stds_by_circle_segment.append(std_density)
    
    #stds_by_circle_segment = np.array(stds_by_circle_segment)
    #for i, value in enumerate(stds_by_circle_segment):
     #   if np.isnan(value):
      #   stds_by_circle_segment[i] = 0
    df_stds_by_circle_segment = pd.DataFrame(columns=['bin', 'standard deviation'])
    df_stds_by_circle_segment['standard deviation'] = stds_by_circle_segment
    df_stds_by_circle_segment['bin'] = df_bins['bin']
    return df_stds_by_circle_segment

def get_coefficient_variation(desired_bin_size, desired_bin_size_angular, df, bool_choice):
    df_bins = create_df_bins(df, desired_bin_size, bool_choice)
    df_stds_by_circle_segment = get_standard_deviation_of_density(desired_bin_size, desired_bin_size_angular, df, bool_choice)
    df_work = get_density(df, desired_bin_size, bool_choice)
    density = df_work['density']
    coefficient_variation = df_stds_by_circle_segment['standard deviation'] / density
    df_coefficient = pd.DataFrame({
        'bin_middle' : df_bins['bin_middle'],
        'coefficient' : coefficient_variation
    })

    return df_coefficient

def extrapolate_first_quadrant_to_circledat(dat):
    dat -= dat.min(axis=0)
    dat = np.vstack([dat, -dat])
    dat = np.vstack([dat, dat * np.array([1, -1])[None, :]])
    dat = np.vstack([dat, dat * np.array([-1, 1])[None, :]])
    return dat
def get_ConvexHull(df, bool_choice):
    is_quarter_circle = bool_choice
    df_ConvexHull = df
    dat = df_ConvexHull[['x', 'y']].values
    if is_quarter_circle:
        dat = extrapolate_first_quadrant_to_circledat(dat)
    else:
        dat = df_ConvexHull[['x', 'y']].values
    hull = ConvexHull(dat)
    
    return hull, df_ConvexHull, dat

def plot_Convex_hull(df, bool_choice):
    is_quarter_circle = bool_choice
    hull, df_ConvexHull, dat = get_ConvexHull(df, bool_choice)
    if is_quarter_circle:
        for simplex in hull.simplices:
            
            plt.plot(df_ConvexHull['x'], df_ConvexHull['y'], 'o', color='blue')
            plt.plot(dat[simplex][:,0], dat[simplex][:,1], 'k-', color='black')
    else:
        for simplex in hull.simplices:
            plt.plot(df_ConvexHull['x'], df_ConvexHull['y'], 'o', color='blue')
            plt.plot(dat[simplex][:,0], dat[simplex][:,1], color='black')
    

    return 
def get_circularity(df, bool_choice):
    hull, df_ConvexHull, dat = get_ConvexHull(df, bool_choice)
    convex_hull_area = hull.volume
    convex_hull_perimeter = hull.area
    Circularity = ((4 * np.pi * (convex_hull_area / convex_hull_perimeter **2)))

    return Circularity
def get_alphashape(df, alpha_scale):
    points = list(zip(df['x'].dropna(), df['y'].dropna()))
    #alpha scale is probably inverted so 1/0.025
    alpha_shape = alphashape.alphashape(points, alpha_scale)

    print(type(alpha_shape), getattr(alpha_shape, "geom_type", None))

    f, ax = plt.subplots()
    ax.scatter(*zip(*points), s=10)

    if alpha_shape is not None and not alpha_shape.is_empty:
        if alpha_shape.geom_type == "Polygon":
            ax.add_patch(
                MplPolygon(alpha_shape.exterior.coords,
                        closed=True,
                        alpha=0.2,
                        edgecolor='black',
                        facecolor='blue')
            )
        elif alpha_shape.geom_type == "MultiPolygon":
            for poly in alpha_shape.geoms:
                ax.add_patch(
                    MplPolygon(poly.exterior.coords,
                            closed=True,
                            alpha=0.2,
                            edgecolor='black',
                            facecolor='blue')
                )
        else:
            print("alpha_shape is not a Polygon/MultiPolygon:", alpha_shape.geom_type)
    else:
        print("alpha_shape is empty or invalid:", alpha_shape)

    plt.show()
    return alpha_shape

def get_bet_centrality(df_edges): ## needs two df one for edges and one for nodes
    df_edges['resistance'] = df_edges['length'] / ((df_edges['width'] / 2) **4)
    edge_list = list(df_edges[['endnodes_1', 'endnodes_2', 'resistance']]. itertuples(index=False, name=None))
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list)
    bet_centrality = nx.betweenness_centrality(G, weight='weight')
    return bet_centrality
def plot_centrality(df_nodes, df_edges, file, node_size, fontsize):
    bet_centrality = get_bet_centrality(df_edges)
    df_nodes['betweenness_centrality'] = df_nodes['name'].map(bet_centrality).fillna(0.0)
    graph = MyceliumGraph(file)
    centrality_values = df_nodes.set_index('name')['betweenness_centrality']
    node_colors_centrality = centrality_values.reindex(graph.G.nodes()).fillna(0.0)

    # use a red color map where low=light and high=red
    cmap = plt.cm.get_cmap('Reds')
    norm = plt.Normalize(vmin=node_colors_centrality.min(), vmax=node_colors_centrality.max())

    f, ax = graph.draw(node_color=node_colors_centrality, cmap=cmap, node_size=node_size)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = f.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label('Betweenness Centrality', fontsize=12)
    ax.set_title('betweenness centrality of every node', fontsize= fontsize)
    return

def Log_centrality(df_nodes, df_edges, file):
    from matplotlib.colors import LogNorm
    graph = MyceliumGraph(file)
    bet_centrality = get_bet_centrality(df_edges)
    df_nodes['betweenness_centrality'] = bet_centrality
    centrality = df_nodes.set_index('name')['betweenness_centrality'].reindex(graph.G.nodes())#.fillna(0.0)
    cmap = plt.cm.get_cmap('Reds')
    positive = centrality > 0
    if positive.any():
        norm = LogNorm(vmin=centrality[positive].min(), vmax=centrality[positive].max())
        colors = np.zeros((len(centrality), 4))
        colors[positive] = cmap(norm(centrality[positive]))
        colors[~positive] = (0.8, 0.8, 0.8, 1.0)   # or another color for zeros
    else:
        colors = np.tile(cmap(0.0), (len(centrality), 1))

    f, ax = graph.draw(node_color=colors, node_size=1, edge_color='black')

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = f.colorbar(sm, ax=ax, fraction=0.03, pad=0.04)
    cb.set_label('Betweenness Centrality', fontsize=12)
    ax.set_title('betweenness centrality of every node in log form', fontsize=14)
    plt.show()  
    return

def get_anastomosis(df_edges):
    df_edges['resistance'] = df_edges['length'] / (df_edges['width'] / 2) **4
    edge_list = list(df_edges[['endnodes_1', 'endnodes_2', 'resistance']]. itertuples(index=False, name=None))
    G = nx.Graph()
    G.add_weighted_edges_from(edge_list, weight='resistance')
    a = nx.minimum_spanning_tree(G, weight='resistance')
    anastomosis = (len(df_edges) / len(a.edges) -1)
    return anastomosis, edge_list

def get_branching_angle(df_nodes, df_edges):
    df_nodes['index'] = df_nodes['name']
    df_nodes = df_nodes.set_index('index')

    df_degree_3 = df_nodes.query('degree == 3')
    df_junctions = df_edges
    df_junctions['endnodes_1_deg'] = df_junctions['endnodes_1'].map(df_nodes['degree'])
    df_junctions['endnodes_2_deg'] = df_junctions['endnodes_2'].map(df_nodes['degree'])
    df_junctions['x_start'] = df_junctions['endnodes_1'].map(df_nodes['x'])
    df_junctions['y_start'] = df_junctions['endnodes_1'].map(df_nodes['y'])
    df_junctions['x_end'] = df_junctions['endnodes_2'].map(df_nodes['x'])
    df_junctions['y_end'] = df_junctions['endnodes_2'].map(df_nodes['y'])

    for i in df_degree_3.index:
    
        df_test = df_junctions.query(f'endnodes_2 == {i} or endnodes_1 == {i}')
        vectors = []
        center = df_degree_3.query(f'name == {i}')
        for n in range(len(df_test)):
            other_end1_name = (df_test['endnodes_1'].iloc[n])
            other_end1 = df_nodes.query(f'name == {other_end1_name}')
            c = center['name'].values
            o = other_end1['name'].values
            if c == o:
                other_end2_name = (df_test['endnodes_2'].iloc[n])
                other_end2 = df_nodes.query(f'name == {other_end2_name}')
                other = other_end2
            else:
                other = other_end1

            vector = (
                other['x'].values - center['x'].values,
                other['y'].values - center['y'].values
                )
            vectors.append(vector)                 
        angles = []

        v1 = np.squeeze(np.array(vectors[0]))
        if v1.any() == 0:
         print(f" vector 1: invalid value at center node {c} with neighbor node {other['name'].values}")
        v2 = np.squeeze(np.array(vectors[1]))
        if v2.any() == 0:
         print(f" vector 2: invalid value at center node {c} with neighbor node {other['name'].values}")
        v3 = np.squeeze(np.array(vectors[2]))
        if v3.any() == 0:
            print(f" vector 3: invalid value at center node {c} with neighbor node {other['name'].values}")

        cosang1 = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        cosang1 = np.clip(cosang1, -1.0, 1.0)
        angle1 = (np.degrees(np.arccos(cosang1)))
        cosang2 = np.dot(v1, v3) / (np.linalg.norm(v1) * np.linalg.norm(v3))
        cosang2 = np.clip(cosang2, -1.0, 1.0)
        angle2 = (np.degrees(np.arccos(cosang2)))
        cosang3 = np.dot(v2, v3) / (np.linalg.norm(v2) * np.linalg.norm(v3))
        cosang3 = np.clip(cosang3, -1.0, 1.0)
        angle3 = (np.degrees(np.arccos(cosang3)))
        angles.append(angle1)
        angles.append(angle2)
        angles.append(angle3)
        branching_angle = np.min(angles)
        df_degree_3.loc[i, 'branching_angle'] = branching_angle
    branching_angle = df_degree_3['branching_angle'].mean()
    return branching_angle


def get_braching_angle_fast(df_nodes, df_edges):
    """Return the mean smallest angle at degree-three junctions.

    This avoids repeatedly filtering the full nodes and edges DataFrames for
    every junction, using coordinate and adjacency lookups built once instead.
    """
    nodes = df_nodes.set_index('name')
    coordinates = nodes[['x', 'y']].to_dict('index')
    adjacency = {}
    for node_1, node_2 in df_edges[['endnodes_1', 'endnodes_2']].to_numpy():
        adjacency.setdefault(node_1, []).append(node_2)
        adjacency.setdefault(node_2, []).append(node_1)

    angles = []
    for node_id in nodes.index[nodes['degree'].eq(3)]:
        neighbors = adjacency.get(node_id, [])
        if len(neighbors) != 3:
            continue
        center = np.array([coordinates[node_id]['x'], coordinates[node_id]['y']])
        vectors = np.array([
            [coordinates[neighbor]['x'], coordinates[neighbor]['y']]
            for neighbor in neighbors
        ]) - center
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms == 0):
            continue

        cosine = np.array([
            np.dot(vectors[0], vectors[1]) / (norms[0] * norms[1]),
            np.dot(vectors[0], vectors[2]) / (norms[0] * norms[2]),
            np.dot(vectors[1], vectors[2]) / (norms[1] * norms[2]),
        ])
        angles.append(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).min())

    return float(np.mean(angles)) if angles else np.nan


def get_branch_and_appical_length(df_edges, df_nodes):
    df_nodes['index'] = df_nodes['name']
    df_nodes = df_nodes.set_index('index')


    df_edges_work = df_edges
    df_edges_work['endnodes_1_deg'] = df_edges_work['endnodes_1'].map(df_nodes['degree'])
    df_edges_work['endnodes_2_deg'] = df_edges_work['endnodes_2'].map(df_nodes['degree'])
    df_edges_work['x_start'] = df_edges_work['endnodes_1'].map(df_nodes['x'])
    df_edges_work['y_start'] = df_edges_work['endnodes_1'].map(df_nodes['y'])
    df_edges_work['x_end'] = df_edges_work['endnodes_2'].map(df_nodes['x'])
    df_edges_work['y_end'] = df_edges_work['endnodes_2'].map(df_nodes['y'])

    df_appical_compartments = df_edges_work.query('endnodes_2_deg == 1 or endnodes_1_deg == 1')
    df_branching = df_edges_work.query('endnodes_1_deg != 1 and endnodes_2_deg != 1')
    length_appical = df_appical_compartments['length'].mean()
    length_branches = df_branching['length'].mean()
    return length_appical, length_branches

def create_df_bins(df_nodes, desired_bin_size, bool_choice):
    true_areas = get_area_of_segments(df_nodes, desired_bin_size, bool_choice)
    count, bin_borders, bin_middle = get_bin_borders_count(df_nodes, desired_bin_size, bool_choice)
    df_bins = pd.DataFrame({'bin_middle': bin_middle, 'bin_count': count, 'bin_area': true_areas})
    # Use the same zero-based bin numbers as ``np.digitize(...)-1`` below.
    # Previously these were one-based, so the coverage of annulus ``i`` was
    # divided by the area of annulus ``i - 1``.
    df_bins['bin'] = np.arange(len(df_bins))
    df_bins['bin_border_min'] = bin_borders[:-1]
    df_bins['bin_border_max'] = bin_borders[1:]
    return df_bins
def area_coverage(file_xlsx, relative_choice, desired_bin_size, bool_choice):
    
    df_nodes, myceg = convert_df_original(file_xlsx)
    df_width = pd.DataFrame(myceg.edges, columns=myceg.edges.columns)
    df_width = df_width.drop(columns=['type', 'name', 'phi'])
    df_bins = create_df_bins(df_nodes, desired_bin_size, bool_choice)
    count, bin_borders, bin_middle = get_bin_borders_count(df_nodes, desired_bin_size, bool_choice)

    coords = df_nodes.set_index('name')[['x', 'y']]
    #transfering x and y coordinates of start and endnode to the df
    df_width['x1'] = df_width['endnodes_1'].map(coords['x'])
    df_width['y1'] = df_width['endnodes_1'].map(coords['y'])
    df_width['x2'] = df_width['endnodes_2'].map(coords['x'])
    df_width['y2'] = df_width['endnodes_2'].map(coords['y'])

    #calculating the area of each edge
    df_width['area'] = (df_width['width'] * df_width['length']) 

    #finding the middle of each edge 
    df_width['edge_middle_x'] = (df_width['x1'] + df_width['x2']) /2
    df_width['edge_middle_y'] = (df_width['y1'] + df_width['y2']) /2

    # Find the radial distance of every edge using the same origin as the
    # node bins.  Quarter-circle measurements use the lower-left node
    # coordinate as their origin (see get_radial_dist).
    edge_x = df_width['edge_middle_x']
    edge_y = df_width['edge_middle_y']
    if bool_choice:
        edge_x = edge_x - df_nodes['x'].min()
        edge_y = edge_y - df_nodes['y'].min()
    df_width['edge_radial_distance'] = np.sqrt(
        edge_x**2 + edge_y**2
    )
    # sorting the edges into the circle segments using the radial distance of their middle point
    df_width['bin'] = np.digitize(df_width['edge_radial_distance'], bin_borders) -1

    df_width = df_width[(df_width['bin'] >= 0) & (df_width['bin'] < len(bin_middle))]

    bin_area = ( df_width.groupby('bin')['area'].sum().reindex(range(len(bin_middle)), fill_value=0))
    if relative_choice == False:
        #plotting
        f, ax = plt.subplots()
        ax.scatter(bin_middle, bin_area)
        ax.set_xlabel('radial distance', fontsize = 14)
        ax.set_ylabel('area covered by edges in bin', fontsize = 14)
        ax.set_title('area coverage by edges', fontsize = 16)
        return df_width, f
    else:
        flaeche = df_bins.set_index('bin')[['bin_area']]
        df_width['bin_area'] = df_width['bin'].map(flaeche['bin_area'])

        df_width['relative_area_covered'] = (df_width['area'] / df_width['bin_area']) 

        relative_area_covered = df_width['relative_area_covered']
        relative_area_covered = ( df_width.groupby('bin')['relative_area_covered'].sum().reindex(range(len(bin_middle)), fill_value=0)) *100

        g, ax = plt.subplots()
        ax.scatter(bin_middle, relative_area_covered)
        ax.set_title('relative area covered by edges', fontsize=16)
        ax.set_xlabel('radial distance', fontsize=14)
        ax.set_ylabel('area coverage in %', fontsize=14)
        plt.show()
        return df_width, g


def area_coverage_geometric(file_xlsx, relative_choice, desired_bin_size, bool_choice):
    """Measure covered annulus area from the union of width-sized edge strips.

    Unlike :func:`area_coverage`, this calculation clips every edge to each
    annulus and unions overlaps before measuring its area.  Consequently, a
    relative coverage cannot exceed 100 percent unless the input geometries or
    their units are invalid.
    """
    df_nodes, myceg = convert_df_original(file_xlsx)
    df_edges = pd.DataFrame(myceg.edges, columns=myceg.edges.columns)
    df_bins = create_df_bins(df_nodes, desired_bin_size, bool_choice)

    coordinates = df_nodes.set_index('name')[['x', 'y']]
    origin_x = df_nodes['x'].min() if bool_choice else 0.0
    origin_y = df_nodes['y'].min() if bool_choice else 0.0
    edge_strips = []
    for edge in df_edges.itertuples(index=False):
        x1, y1 = coordinates.loc[edge.endnodes_1]
        x2, y2 = coordinates.loc[edge.endnodes_2]
        edge_strips.append(
            LineString([(x1, y1), (x2, y2)]).buffer(
                edge.width / 2,
                cap_style='flat',
            )
        )
    covered_geometry = unary_union(edge_strips)

    covered_areas = []
    for bin_row in df_bins.itertuples(index=False):
        outer = Point(origin_x, origin_y).buffer(bin_row.bin_border_max, quad_segs=64)
        inner = Point(origin_x, origin_y).buffer(bin_row.bin_border_min, quad_segs=64)
        annulus = outer.difference(inner)
        if bool_choice:
            annulus = annulus.intersection(
                box(origin_x, origin_y, origin_x + bin_row.bin_border_max,
                    origin_y + bin_row.bin_border_max)
            )
        covered_areas.append(covered_geometry.intersection(annulus).area)

    df_bins = df_bins.copy()
    df_bins['area_covered'] = covered_areas
    df_bins['relative_area_covered'] = 100 * df_bins['area_covered'] / df_bins['bin_area']

    f, ax = plt.subplots()
    if relative_choice:
        ax.scatter(df_bins['bin_middle'], df_bins['relative_area_covered'])
        ax.set_title('geometric relative area covered by edges', fontsize=16)
        ax.set_ylabel('area coverage in %', fontsize=14)
    else:
        ax.scatter(df_bins['bin_middle'], df_bins['area_covered'])
        ax.set_title('geometric area coverage by edges', fontsize=16)
        ax.set_ylabel('area covered by edges in bin', fontsize=14)
    ax.set_xlabel('radial distance in µm', fontsize=14)
    return df_bins, f

def phenotype_deviation(file_gen, file_orig):
    desired_bin_size=35
    desired_bin_size_angular = 30
    ## loading the data from the original fungus and the generated one ##
    df_generated, myceg_simu = convert_df_generated(file_gen)
    df_edges_generated = pd.DataFrame(myceg_simu.edges, columns=myceg_simu.edges.columns)
    df_original, myceg = convert_df_original(file_orig)
    df_edges_original = pd.DataFrame(myceg.edges, columns=myceg.edges.columns)
    ## calculating the different meassures for the original and the generated #
    #for the original
    df_density_orig = get_density(df_original, desired_bin_size=desired_bin_size, bool_choice=True)
    circularity_orig = get_circularity(df_original, True)
    anastomosis_orig, c = get_anastomosis(df_edges_original)
    std_orig = get_standard_deviation_of_density(desired_bin_size=desired_bin_size, desired_bin_size_angular=desired_bin_size_angular, df=df_original, bool_choice=True)
    std_orig = np.mean(std_orig)
    length_appical_orig, length_branches_orig = get_branch_and_appical_length(df_edges_original, df_original)
    branching_angle_orig = get_braching_angle_fast(df_original, df_edges_original)
    # for the generated
    df_density_gen = get_density(df_generated, desired_bin_size=desired_bin_size, bool_choice=False)
    circularity_gen = get_circularity(df_generated, False)
    anastomosis_gen, d = get_anastomosis(df_edges_generated)
    std_gen = get_standard_deviation_of_density(desired_bin_size=desired_bin_size, desired_bin_size_angular=desired_bin_size_angular, df=df_generated, bool_choice=False)
    std_gen = np.mean(std_gen)
    length_appical_gen, length_branches_gen = get_branch_and_appical_length(df_edges_generated, df_generated)
    branching_angle_gen = get_braching_angle_fast(df_generated, df_edges_generated)
    ## focusing the density so the wavefront and the inoculum are not counted (focus on the plateau) ##
    plateau_threshold_lower_orig = df_density_orig['density'].median() - df_density_orig['density'].median() / 2
    plateau_threshold_upper_orig = df_density_orig['density'].median() + df_density_orig['density'].median() /2
    plateau_threshold_lower_gen = df_density_gen['density'].median() - df_density_gen['density'].median() /2
    plateau_threshold_upper_gen = df_density_gen['density'].median() + df_density_gen['density'].median() /2
    df_plateau_orig = df_density_orig.query(f"density >= {plateau_threshold_lower_orig} and density <= {plateau_threshold_upper_orig}")
    df_plateau_gen = df_density_gen.query(f"density >= {plateau_threshold_lower_gen} and density <= {plateau_threshold_upper_gen}")
    density_orig = df_plateau_orig['density'].mean()
    density_gen = df_plateau_gen['density'].mean()

    ## computing the actual density difference and not the relative error
    delta_density = (density_orig - density_gen) / (density_orig)

    
    ## calculating the relative error to get a values for similarity ##

    dev_density = (abs(density_gen - density_orig) / abs(density_orig))
    dev_standard_div = (abs(std_gen - std_orig) / abs(std_orig))
    dev_circularity = (abs(circularity_gen - circularity_orig) / abs(circularity_orig))
    #dev_anastomosis = (abs(anastomosis_orig - anastomosis_gen ) / abs(anastomosis_orig))
    dev_edge_length_appical = (abs(length_appical_gen - length_appical_orig) / abs(length_appical_orig))
    dev_edge_length_branch =  (abs(length_branches_gen - length_branches_orig) / abs(length_branches_orig))
    dev_branching_angle = (abs(branching_angle_gen - branching_angle_orig) / abs(branching_angle_orig))
    #D = 1/4 *((dev_density) + (dev_standard_div) + (dev_circularity) + (dev_anastomosis))
    
    #phenotype_deviation1 = abs(dev_density) + abs(dev_circularity) + abs(dev_anastomosis) + abs(dev_standard_div)
    phenotype_deviation = [dev_density, dev_standard_div, dev_circularity, delta_density, dev_edge_length_appical, dev_edge_length_branch, dev_branching_angle]
    return phenotype_deviation
