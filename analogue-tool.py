import georasters as gr
import pandas as pd
import math

PREC_PATH = 'wc2.1_10m_prec/wc2.1_10m_prec_{}.tif'
TAVG_PATH = 'wc2.1_10m_tavg/wc2.1_10m_tavg_{}.tif'
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
    assert isinstance(ref, Site), "Must be a site objet with longitude and latitude"
    assert isinstance(weight, tuple), "!!!"

    dissimilarity = 0.0
    all_dissimilarities = []
    path = build_path(PREC_PATH, 1)
    precs = gr.from_file(path).to_pandas()

    for j in range(num_site):
        target_ = precs.loc[j]
        target = Site(target_.x, target_.y)

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

    return all_dissimilarities


ref = Site(-75.5, 3.2)
target = Site(-78.5, -89.83333333333331)

# print(ccafs(ref, target, season=2, weight=(0.5, 0.5), z=2))
print(ccafs_all(ref, season=2, num_site=2, weight=(0.5, 0.5), z=2))

