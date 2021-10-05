import georasters as gr
import pandas as pd
import math
import pymp
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD
nprocs = comm.Get_size()
rank = comm.Get_rank()

PREC_PATH = '/opt/Agriculture/wc2.1_10m_prec/wc2.1_10m_prec_{}.tif'
TAVG_PATH = '/opt/Agriculture/wc2.1_10m_tavg/wc2.1_10m_tavg_{}.tif'
TMIN_PATH = ''
TMAX_PATH = ''

REFERENCE_F = ''
TAGET_P = ''


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


def p_ccafs_all(ref, season, num_site, weight, z):
    comm = MPI.COMM_WORLD
    nprocs = comm.Get_size()
    rank = comm.Get_rank()
    
    PREC_PATH = '/opt/Agriculture/wc2.1_10m_prec/wc2.1_10m_prec_{}.tif'
    TAVG_PATH = '/opt/Agriculture/wc2.1_10m_tavg/wc2.1_10m_tavg_{}.tif'
    
    start_time = time.time()
    assert isinstance(ref, Site), "Must be a site objet with longitude and latitude"
    assert isinstance(weight, tuple), "!!!"

    all_dissimilarities = []
    path = build_path(PREC_PATH, 1)
    precs = gr.from_file(path).to_pandas()

    # print(precs.head().y)

    list_of_sites = [val for val in range(num_site)]
    my_site = split_data(list_of_sites, nprocs, rank)

    # print(rank, my_site)
    for j in my_site:
        target_ = precs.loc[j]
        target = Site(target_.x, target_.y)
        dissimilarity = 0.0
        diss = pymp.shared.list()
        with pymp.Parallel(season) as p:
            if not (target.longitude == ref.longitude or target.latitude == ref.latitude):
                for i in p.range(1, season + 1):
                    path_prec = build_path(PREC_PATH, i)
                    path_tavg = build_path(TAVG_PATH, i)

                    prec = gr.from_file(path_prec).to_pandas()
                    tavg = gr.from_file(path_tavg).to_pandas()

                    if prec.keys()[2] == True and tavg.keys()[2] == True:
                        prec.columns = ['row', 'col', 'value', 'x', 'y']
                        tavg.columns = ['row', 'col', 'value', 'x', 'y']

                    ref_prec = extract(prec, ref)
                    target_prec = extract(prec, target)

                    ref_tavg = extract(tavg, ref)
                    target_tavg = extract(tavg, target)

                    # print(ref_tavg.value, target_tavg.value)
                    # print(ref_prec.value, target_prec.value)

                    diss.append((weight[0] * math.pow((ref_tavg.value - target_tavg.value), z)) + (
                            weight[1] * math.pow((ref_prec.value - target_prec.value), z)))

                all_dissimilarities.append((target.longitude, target.latitude, math.pow(sum(diss), (1 / z))))

    if rank != 0:
        comm.send(all_dissimilarities, dest=0, tag=rank)

    else:
        results = [item for item in all_dissimilarities]
        for rk in range(1, nprocs):
            resp = comm.recv(source=rk, tag=rk)
            results += [elt for elt in resp]

    end_time = time.time() - start_time
    print(rank, " The computing process took : ", end_time)

    if rank == 0:
        print(results)
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


#ref = Site(-75.5, 3.2)
# target = Site(-78.5, -89.83333333333331)

# print(ccafs(ref, target, season=2, weight=(0.5, 0.5), z=2))
# print(ccafs_all(ref, season=2, num_site=1, weight=(0.5, 0.5), z=2))
# print(parallel_ccafs_all(ref, season=2, num_site=4, weight=(0.5, 0.5), z=2, num_threads=4))
#p_ccafs_all(ref, season=2, num_site=10, weight=(0.5, 0.5), z=2)

def get_similarity(lat, long, refs_array):
    # refs_array = [[2.343 5.231] [3.233 5.234]]
    reference = Site(lat, long)
    return p_ccafs_all(reference, season=2, num_site=2, weight=(0.5, 0.5), z=2)
