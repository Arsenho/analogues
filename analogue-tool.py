import csv

import georasters as gr
import pandas as pd
import math
import pymp
import time
import re
import importlib.machinery, importlib.util
import os
from geopandas import GeoDataFrame
from shapely.geometry import Point, Polygon
from mpi4py import MPI
import json

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

global PREC_PATH, TAVG_PATH, TMIN_PATH, TMAX_PATH, REFERENCE_F, TAGET_P
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

PREC_PATH = os.path.join(BASE_DIR, 'wc2.1_10m_prec/wc2.1_10m_prec_{}.tif')
TAVG_PATH = os.path.join(BASE_DIR, 'wc2.1_10m_tavg/wc2.1_10m_tavg_{}.tif')
TMIN_PATH = os.path.join(BASE_DIR, 'wc2.1_10m_tavg/wc2.1_10m_tmin_{}.tif')
TMAX_PATH = os.path.join(BASE_DIR, 'wc2.1_10m_tavg/wc2.1_10m_tmax_{}.tif')

AFRICA_PREC_PATH = os.path.join(BASE_DIR, 'wc2.1_10m_prec/africa')
AFRICA_TAVG_PATH = os.path.join(BASE_DIR, 'wc2.1_10m_tavg/africa')

RESULTS_PATH = os.path.join(BASE_DIR, 'results')

REFERENCE_F = ''
TAGET_P = ''

PATHS = [PREC_PATH, TAVG_PATH, TMIN_PATH, TMAX_PATH]

africa_cords = [(27.62901, -17.709524),
                (10.081337, -20.873587),
                (-6.026967, -3.998587),
                (-13.641743, 5.845163),
                (-22.000126, 7.954538),
                (-31.105649, 12.173288),
                (-35.796452, 18.149851),
                (-38.596287, 29.048288),
                (-35.224105, 36.431101),
                (-27.738019, 48.384226),
                (-19.700546, 56.118601),
                (-6.376472, 51.548288),
                (8.346274, 56.118601),
                (14.204722, 47.329538),
                (29.481663, 36.079538),
                (35.696542, 26.235788),
                (39.048178, 10.767038),
                (38.224351, -1.537649),
                (34.256286, -11.732962),
                (27.317094, -18.764212)]

africa_cords_list = []

africa_polygon = Polygon(africa_cords)


def filter_csvfile(name):
    return re.match("^\w+.csv$", name)


def check_point_within_africa(lat, long):
    current_point = Point(lat, long)
    return current_point.within(africa_polygon)


def build_path(path, mth):
    if mth in range(1, 10):
        return str(path).format('0' + str(mth))
    else:
        return str(path).format(str(mth))


class Site:
    def __init__(self, x, y):
        self.longitude = x
        self.latitude = y


def dtr(val):
    return 0


def extract(var, var_to_extract):
    assert isinstance(var, pd.DataFrame)
    assert isinstance(var_to_extract, Site)
    indx = 0
    for i in range(len(var.value)):
        if round(var.x[i], 1) == round(var_to_extract.longitude, 1) and round(var.y[i], 1) == round(
                var_to_extract.latitude, 1):
            indx = i

    return var.loc[indx]


def ccafs(ref, target, season, weight, z):
    assert isinstance(ref, Site), "Must be a site objet with longitude and latitude"
    assert isinstance(target, Site), "Must be a site objet with longitude and latitude"
    assert isinstance(weight, tuple)

    dissimilarity = 0.0

    for i in range(1, season + 1):
        path_prec = build_path(PREC_PATH, i)
        path_tavg = build_path(TAVG_PATH, i)

        prec = gr.from_file(path_prec).to_pandas()
        tavg = gr.from_file(path_tavg).to_pandas()

        ref_prec = extract(prec, ref)
        target_prec = extract(prec, target)

        ref_tavg = extract(tavg, ref)
        target_tavg = extract(tavg, target)

        # print(ref_tavg.value, target_tavg.value)
        # print(ref_prec.value, target_prec.value)

        dissimilarity += (weight[0] * math.pow((ref_tavg.value - target_tavg.value), z)) + (
                weight[1] * math.pow((ref_prec.value - target_prec.value), z))

    # for cpt in range(1, 13):
    #     temp = dtr('f') / dtr('p')

    return math.pow(dissimilarity, (1 / z))


def ccafs_all(ref, season, num_site, weight, z):
    start_time = time.time()
    assert isinstance(ref, Site), "Must be a site objet with longitude and latitude"
    assert isinstance(weight, tuple), "!!!"

    all_dissimilarities = []
    path = build_path(PREC_PATH, 1)
    precs = gr.from_file(path).to_pandas()

    for j in range(num_site):
        target_ = precs.loc[j]
        target = Site(target_.x, target_.y)
        dissimilarity = 0.0
        if not (target.longitude == ref.longitude or target.latitude == ref.latitude):
            for i in range(1, season + 1):
                path_prec = build_path(PREC_PATH, i)
                path_tavg = build_path(TAVG_PATH, i)

                prec = gr.from_file(path_prec).to_pandas()
                tavg = gr.from_file(path_tavg).to_pandas()

                ref_prec = extract(prec, ref)
                target_prec = extract(prec, target)

                ref_tavg = extract(tavg, ref)
                target_tavg = extract(tavg, target)

                # print(ref_tavg.value, target_tavg.value)
                # print(ref_prec.value, target_prec.value)

                dissimilarity += (weight[0] * math.pow((ref_tavg.value - target_tavg.value), z)) + (
                        weight[1] * math.pow((ref_prec.value - target_prec.value), z))

            all_dissimilarities.append((target.longitude, target.latitude, math.pow(dissimilarity, (1 / z))))

    end_time = time.time() - start_time
    print(end_time)
    return all_dissimilarities


def split_data(data, nprocs, rank):
    r, diff = divmod(len(data), nprocs)
    counts = [r + 1 if p < diff else r for p in range(nprocs)]

    # determine the starting and ending indices of each sub-task
    starts = [sum(counts[:p]) for p in range(nprocs)]
    ends = [sum(counts[:p + 1]) for p in range(nprocs)]

    # converts data into a list of arrays
    datas = [data[starts[p]:ends[p]] for p in range(nprocs)]

    return datas[rank]


def exists_africa(paths):
    assert isinstance(paths, tuple), ""
    cpt = 0
    for path in paths:
        if str(path).split(".")[-1] == 'csv':
            cpt += 1
    if cpt != len(paths):
        return False

    return True


def p_write_current_result(file, current_result):
    assert isinstance(file, str), "give a valid file path"
    assert isinstance(current_result, list), "the data to write must be a list of items"
    with open(file, "a") as result_file:
        writer = csv.writer(result_file)
        writer.writerow(current_result)


def p_ccafs_all(ref, season, weight, z, sites="africa"):
    start_time = time.time()
    if rank == 0:
        os.system("clear")
        print("\t\t ==== Compute similarity for a given site ==== \n\n")

    assert isinstance(ref, Site), "Error : Must be a site objet with longitude and latitude"
    assert isinstance(weight, tuple), "!!!"

    all_dissimilarities = []

    if sites == "africa":
        prec_path = AFRICA_PREC_PATH
        tavg_path = AFRICA_TAVG_PATH
    else:
        prec_path = PREC_PATH
        tavg_path = TAVG_PATH

    files_prec = [os.path.join(BASE_DIR, prec_path, f) for f in os.listdir(prec_path) if
                  os.path.isfile(os.path.join(prec_path, f))]
    files_tavg = [os.path.join(BASE_DIR, tavg_path, f) for f in os.listdir(tavg_path) if
                  os.path.isfile(os.path.join(tavg_path, f))]

    if rank == 0:
        print("\t\t==== Collecting data for computation --- OK ==== ")

    # print(files_prec[-1])
    # print(files_tavg[-1])

    precs = pd.read_csv(files_prec[0])
    num_site = len(precs.value)

    # print(num_site)
    # print(precs.head().y)

    list_of_sites = [val for val in range(num_site)]
    my_site = split_data(list_of_sites, nprocs, rank)

    # print(rank, my_site)
    if rank == 0:
        print("\t\t==== Start computations ==== ")

    for j in my_site:
        target_ = precs.loc[j]
        target = Site(target_.x, target_.y)
        diss = pymp.shared.list()

        if not (target.longitude == ref.longitude or target.latitude == ref.latitude):
            if sites == "africa":
                with pymp.Parallel(season) as p:
                    for i in p.range(len(files_tavg)):
                        prec = pd.read_csv(files_prec[i])
                        tavg = pd.read_csv(files_tavg[i])

                        ref_prec = extract(prec, ref)
                        target_prec = extract(prec, target)

                        ref_tavg = extract(tavg, ref)
                        target_tavg = extract(tavg, target)

                        # print(ref_tavg.value, target_tavg.value)
                        # print(ref_prec.value, target_prec.value)

                        diss.append((weight[0] * math.pow((ref_tavg.value - target_tavg.value), z)) + (
                                weight[1] * math.pow((ref_prec.value - target_prec.value), z)))

            else:
                with pymp.Parallel(season) as p:
                    for i in p.range(len(files_tavg)):
                        prec = gr.from_file(files_prec[i]).to_pandas()
                        tavg = gr.from_file(files_tavg[i]).to_pandas()

                        ref_prec = extract(prec, ref)
                        target_prec = extract(prec, target)

                        ref_tavg = extract(tavg, ref)
                        target_tavg = extract(tavg, target)

                        # print(ref_tavg.value, target_tavg.value)
                        # print(ref_prec.value, target_prec.value)

                        diss.append((weight[0] * math.pow((ref_tavg.value - target_tavg.value), z)) + (
                                weight[1] * math.pow((ref_prec.value - target_prec.value), z)))

            current_diss = math.pow(sum(diss), (1 / z))
            current_result_to_save = [target.longitude, target.latitude, current_diss]

            p_write_current_result(os.path.join(RESULTS_PATH, '{}_results.csv'.format(rank)),
                                   current_result_to_save)
            all_dissimilarities.append(
                {
                    'longitude': target.longitude,
                    'latitude': target.latitude,
                    'value': current_diss
                }
                # (target.longitude, target.latitude, current_diss))
            )
        else:
            pass

    if rank == 0:
        print("\t\t==== End computations ==== ")

    if rank != 0:
        comm.send(all_dissimilarities, dest=0, tag=rank)
        del all_dissimilarities
    else:
        results = [item for item in all_dissimilarities]
        for rk in range(1, nprocs):
            resp = comm.recv(source=rk, tag=rk)
            results += [elt for elt in resp]
        del all_dissimilarities
    if rank == 0:
        end_time = time.time() - start_time
        print("\t\t==== Computation execution time --- {} s ==== ".format(end_time))
        print("\t\t==== Computation results --- ==== ")
        print("\t\t", results)
        return results


def parallel_ccafs_all(ref, season, num_site, weight, z, num_threads=4):
    start_time = time.time()
    assert isinstance(ref, Site), "Must be a site objet with longitude and latitude"
    assert isinstance(weight, tuple), "!!!"

    path = build_path(PREC_PATH, 1)
    precs = gr.from_file(path).to_pandas()

    all_dissimilarities = pymp.shared.array(num_site)
    diss = pymp.shared.array(season)

    with pymp.Parallel(num_threads) as p:
        for j in p.range(0, num_site):
            target_ = precs.loc[j]
            target = Site(target_.x, target_.y)
            dissimilarity = 0.0
            if not (target.longitude == ref.longitude or target.latitude == ref.latitude):
                for i in range(1, season + 1):
                    path_prec = build_path(PREC_PATH, i)
                    path_tavg = build_path(TAVG_PATH, i)

                    prec = gr.from_file(path_prec).to_pandas()
                    tavg = gr.from_file(path_tavg).to_pandas()

                    ref_prec = extract(prec, ref)
                    target_prec = extract(prec, target)

                    ref_tavg = extract(tavg, ref)
                    target_tavg = extract(tavg, target)

                    # print(ref_tavg.value, target_tavg.value)
                    # print(ref_prec.value, target_prec.value)

                    dissimilarity += (weight[0] * math.pow((ref_tavg.value - target_tavg.value), z)) + (
                            weight[1] * math.pow((ref_prec.value - target_prec.value), z))

                all_dissimilarities[j] = math.pow(dissimilarity, (1 / z))

    end_time = time.time() - start_time
    print("The computing process took : ", end_time)
    return all_dissimilarities


def set_res(results, point):
    assert isinstance(results, dict)

    results['row'].append(point.row)
    results['col'].append(point.col)
    results['value'].append(point.value)
    results['x'].append(point.x)
    results['y'].append(point.y)

    return results


def get_sub_refs(targets, wc_path, items='all'):
    assert isinstance(targets, list), "The argument must be an array of sites"
    # results = {'row': [], 'col': [], 'value': [], 'x': [], 'y': []}
    # results = pymp.shared.dict({'row': [], 'col': [], 'value': [], 'x': [], 'y': []})
    # results = {'row': [], 'col': [], 'value': [], 'x': [], 'y': []}

    all_dataframes = pymp.shared.list()

    if items == 'all':
        num_threads = 12
    else:
        num_threads = int(items)

    with pymp.Parallel(num_threads) as p:
        for i in p.range(1, 3):
            path = build_path(wc_path, i)
            all_targets = gr.from_file(path).to_pandas()
            all_targets.columns = ['row', 'col', 'value', 'x', 'y']
            results = {'value': [], 'x': [], 'y': []}

            for j in p.range(len(targets)):
                # assert isinstance(targ, list), "Make sure item have their longitude and latitude values"
                site = Site(targets[j][0], targets[j][1])
                targ = extract(all_targets, site)
                targ.columns = ['row', 'col', 'value', 'x', 'y']
                results = set_res(results, targ)

            all_dataframes.append(pd.DataFrame(results))
    return all_dataframes


def get_africa_refs(wc_path, directory='', items='all'):
    # all_dataframes = pymp.shared.list()
    all_dataframes = []
    if items == 'all':
        num_threads = 12
    else:
        num_threads = int(items)

    for i in range(1, 13):
        path = build_path(wc_path, i)
        all_targets = gr.from_file(path).to_pandas()
        all_targets.columns = ['row', 'col', 'value', 'x', 'y']

        print("all -> {}".format(len(all_targets.row)))
        results = {'row': [], 'col': [], 'value': [], 'x': [], 'y': []}

        for j in range(len(all_targets)):
            point = all_targets.loc[j]

            if check_point_within_africa(point.x, point.y):
                inter = set_res(results, point)
                results = inter

        new_path = str(path.split('/')[-1]).split('.')
        file = "{}/{}.{}_africa".format(directory, new_path[0], new_path[1])

        print("file name -> " + file)
        print("item -> {}".format(len(results["row"])))

        df = pd.DataFrame(results)
        df.to_csv(file + ".csv")

        # tif = gr.from_pandas(df)
        # tif.to_tiff(file + ".tif")

        # crs = {'init': 'epsg:4326'}
        # geometry = [Point(xy) for xy in zip(df.x, df.y)]
        # df = df.drop(['x', 'y'], axis=1)
        # gdf = GeoDataFrame(df, crs=crs, geometry=geometry)
        # gdf.to_file(file + ".tif")

        # all_dataframes.append(pd.DataFrame(results))
    # return all_dataframes


def apply_to_files(all_paths):
    print("------ Extraction Started ------")
    for path in all_paths:
        directory = path.split('/')[0]
        res = os.system("mkdir {}/{}".format(directory, "africa"))
        directory = "{}/{}".format(directory, "africa")
        get_africa_refs(path, directory=directory)
    print("------ Extraction Ended ------")


ref_of_dschang = Site(5.4527263, 10.0268688)
# target = Site(-78.5, -89.83333333333331x)

# print(ccafs(ref, target, season=2, weight=(0.5, 0.5), z=2))
# print(ccafs_all(ref, season=2, num_site=1, weight=(0.5, 0.5), z=2))
# print(parallel_ccafs_all(ref, season=2, num_site=4, weight=(0.5, 0.5), z=2, num_threads=4))
# p_ccafs_all(ref, season=2, num_site=2, weight=(0.5, 0.5), z=2)

# print(PREC_PATH)
# (get_sub_refs([[-75.5, 3.2], [-78.5, -89.83333333333331]], PREC_PATH, 2))
# get_africa_refs(PREC_PATH, directory='wc2.1_10m_prec', items=2)
# print(get_sub_refs([[-75.5, 3.2], [-78.5, -89.83333333333331]], TAVG_PATH))

# apply_to_files(PATHS)
p_ccafs_all(ref_of_dschang, season=2, weight=(0.5, 0.5), z=2, sites="africa")
