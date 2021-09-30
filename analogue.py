import math as mt
import math
import numpy as np
import pandas as pd

# The data files
global climstack
global cmip5_table
global sim_index_table

class Parameter:
    def __init__(self, x, y, vars, weights, ndivisions, env_data_ref, env_data_targ,
                 growing_season, rotation, threshold, outfile, fname, writefile):
        """
        :param x -> latitude (decimal degrees)
        :param y -> longitude (decimal degrees)
        :param vars -> character
        vector: a vector with the name of the climatic variable(s) to use, e.g. c("prec","tmean"), or bioclimatic
        variable e.g. "bio_1"
        :param weights -> numeric vector: vector of length equal to the number of variables.
        Each value in the vector gives the weight given to each variable in the range 0-1. The sum of the weights
        must equal 1.
        :param ndivisions -> numeric vector: the number of divisions (usually months) for each
        variable. ndivisions=12 for climatic variables and ndivisions=1 for bioclimatic (or other types of variables)
        variables.
        :param env.data.ref -> list: a list of length equal to the number of variables that specifies the
        reference climatic conditions. Each element in the list is either a RasterLayer or a RasterStack object.
        RasterLayer applies to bioclimatic variables, whereas RasterStack applies for monthly data.
        :param env.data.targ -> list: a list of length equal to the number of variables that specifies the target climatic
        conditions. Each element in the list is either a RasterLayer or a RasterStack object. RasterLayer applies to
        bioclimatic variables, whereas RasterStack applies for monthly data.
        :param growing.season -> numeric vector: growing season (months) of interest in the analysis. Specified as a vector of length 2, where the first value
        specifies the start and the second value specifies the end of growing season. Not relevant for bioclimatic
        variables
        :param rotation -> character: should a rotation be applied. i.e. "tmean", "prec", "both" or "none".
        Rotation will allow comparisons between sites with different seasonality (e.g. northern vs. southern
        hemisphere)
        :param threshold -> numeric: value between 0-1. Only sites with a climatic similarity above this
        threshold will be saved and displayed.
        :param outfile -> character: directory where the resultant similarity map will be saved
        :param fname -> character: name of output file
        :param writefile -> logical: if the output file is to be written on disk. Otherwise only an object will be returned.


        :rtype json

        example: params = createParameters(x=-75.5, y=3.2, vars=c("prec","tmean"),weights=c(0.5,0.5),
                        ndivisions=c(12,12),growing.season=c(1,12),rotation="tmean",threshold=1,
                        env.data.ref=list(prec,tavg), env.data.targ=list(prec,tavg),
                        outfile="~/.",fname='similarity',writefile=FALSE)

        """

        assert isinstance(growing_season, tuple), "Please make sure the growing_season is a tuple"

        self.latitude = x
        self.longitude = y
        self.vars = vars
        self.weights = weights
        self.ndivisions = ndivisions
        self.env_data_ref = env_data_ref
        self.env_data_targ = env_data_targ
        self.growing_season = growing_season
        self.rotation = rotation
        self.threshold = threshold
        self.outfile = outfile
        self.fname = fname
        self.writefile = writefile

    def get_parameter(self):

        if len(self.growing_season) == 1:
            self.growing_season = self.growing_season
        elif len(self.growing_season) == 2 and (self.growing_season[1] == self.growing_season[2]):
            self.growing_season = self.growing_season[1]
        elif len(self.growing_season) == 2 and (self.growing_season[1] > self.growing_season[2]):
            self.growing_season = (self.growing_season[1], 12, 1, self.growing_season[2])
        elif len(self.growing_season) == 2 and (self.growing_season[1] < self.growing_season[2]):
            self.growing_season = (self.growing_season[1], self.growing_season[2])
        elif len(self.growing_season) == 4 and (self.growing_season[1] > self.growing_season[2]):
            self.growing_season = (
                self.growing_season[1], 12, 1, self.growing_season[2], self.growing_season[3], self.growing_season[4])
        elif len(self.growing_season) == 4 and (self.growing_season[3] > self.growing_season[4]):
            self.growing_season = (
                self.growing_season[1], self.growing.season[2], self.growing.season[3], 12, 1, self.growing_season[4])
        else:
            self.growing_season = (
                self.growing_season[1], self.growing_season[2], self.growing_season[3], self.growing_season[4])

        # self.growing_season = unique(self.growing_season)
        return [self.latitude, self.longitude, self.vars, self.weights, self.ndivisions, self.env_data_ref,
                self.env_data_targ, self.growing_season, self.rotation, self.outfile, self.fname, self.writefile]

def extract(firt_data, second_data):
    pass

def cbind(first_value, second_value):
    pass


def sim_index(dist, var):
    assert isinstance(sim_index_table, list)
    k = sim_index_table[:2] # [which(var == sim_index_table$var), 2]
    if len(k) == 0:
        k = 10  # default

    return k / (k + dist)


def rowMeans(res_month, na_rm=False):
    pass


def rowSums(var_sim):
    pass


class Analogue:
    def __init__(self, params):
        assert isinstance(params, Parameter), "Make sure params is well set"
        self.params = params

    def ref_vals(self):
        if isinstance(self.params.env_data_targ[1][1], "RasterStack") or isinstance(self.params.env_data_targ[1][1], "RasterLayer"):
            training_targ = self.params.env_data_targ
        else:
            exit("data in env.data.targ not a RasterStack or RasterLayer")

        val_ref = list()
        for i in range(1, len(self.params.vars)):
            val_ref[i] = int(extract(training_targ[i]), cbind(self.params.latitude, self.params.longitude))

        return val_ref

    def calc_similarity(self):
        val_ref = self.ref_vals()
        if isinstance(self.params.env_data_ref[1][1], "RasterStack") or isinstance(self.params.env_data_ref[1][1], "RasterLayer"):
            training_targ = self.params.env_data_targ
        else:
            exit("data in env.data.targ not a RasterStack or RasterLayer")
        if self.params.rotation is None:
            training_targ = training_targ
        else:
            res = self.similarity(training_targ, val_ref)

        return res

        # simresult = wrapper()
        # names(simresult) < - params$fname
        # if (params
        # $writefile) writeRaster(simresult, paste(params$outfile, "/out_", params$fname, ".tif", sep = ""), overwrite = TRUE)
        # return (simresult)

    def similarity(self, training_targ, val_ref):
        var_sim = np.ndarray((len(training_targ[1])*len(training_targ[1][0]), len(self.params.vars)), dtype=float)
        for i in range(1, len(self.params.vars)):
            result = [elt for elt in range(1, len(training_targ[i]))]
            if len(val_ref[i]) == 1:
                result = [(training_targ[i] - val_ref[i])^2 for cpt in range(len(result))]
                result = [sim_index(math.sqrt(result[cpt]), self.params.vars[i]) for cpt in range(len(result))]
                for j in range(len(var_sim)):
                    var_sim[j][i] =result * self.params.weights[i]
            else:
                res_month = np.ndarray((len(training_targ[1])*len(training_targ[1][0]), 12), dtype=int)
                for k in range(len(res_month)):
                    res_month[k][self.params.growing_season]= (np.transpose(np.transpose(training_targ[i][self.params.growing_season]) - val_ref[i][self.params.growing_season]))^2
                
                result = [sim_index(math.sqrt(rowMeans(res_month, na_rm = True)), self.params.vars[i]) for cpt in range(len(result))]

                for k in range(len(var_sim)):
                    var_sim[k][i]=result * self.params.weights[i]

        combined = training_targ[i][1]
        if len(self.params.vars) > 1:
            combined = [round(rowSums(var_sim), digits=3) for cpt in range(len(var_sim))]

        else:
            pass
            # combined = [round(var_sim[], digits=3) for cpt in range(len(var_sim))]

        return combined


precipitation = gr.from_file("precipitation.tif")

precipitation = precipitation.to_pandas()

precipitation.head()