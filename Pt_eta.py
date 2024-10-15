# %%
import numpy as np
import json
import awkward as ak
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea import processor
import hist
from pathlib import Path
from hist import Hist
import pandas as pd
import coffea
import argparse
import os
import shutil


# %%
class MyProcessor(processor.ProcessorABC):
    def __init__(self):
        pass
    def process(self, events):
        dataset = events.metadata["dataset"]

        results = {}
        results[dataset] = {
            "count": len(events)
        }
        #pT
        h_photon_EB_pt = (
            Hist.new.StrCat([], growth=True, name="dataset_photon_EB_pt", label="Dataset")
            .Reg(40, 0, 5000, overflow=False, underflow=False, name="x", label = "[Gev]" )
            .Weight()
        )
        h_photon_EE_pt = (
            Hist.new.StrCat([], growth=True, name="dataset_photon_EE_pt", label="Dataset")
            .Reg(40, 0, 5000, overflow=False, underflow=False, name="x", label = "[Gev]" )
            .Weight()
        )
        #eta
        h_photon_EB_eta = (
            Hist.new
            .StrCat([], growth=True, name="dataset_photon_EB_eta", label="dataset")
            .Reg(40, -2.5, 2.5, overflow=False, underflow=False, name="x", label = "eta")
            .Weight()
        )
        h_photon_EE_eta = (
            Hist.new
            .StrCat([], growth=True, name="dataset_photon_EE_eta", label="dataset")
            .Reg(40, -2.5, 2.5, overflow=False, underflow=False, name="x", label = "eta")
            .Weight()
        )
        # separate the events into EB and EE
        photon_EB = events.Photon[events.Photon.isScEtaEB]
        photon_EE = events.Photon[events.Photon.isScEtaEE]

        photon_EB_no_none = photon_EB[~ak.is_none(photon_EB, axis=1)]      
        photon_EE_no_none = photon_EE[~ak.is_none(photon_EE, axis=1)]      
        
        # EB_total_selection = ak.fill_none(
        #     ak.num(photon_EB_selected,axis=1) > 1,
        #     False
        # )

        # EE_total_selection = ak.fill_none(
        #     ak.num(photon_EE,axis=1) > 1,
        #     False
        # )

        EB_photon_selection = photon_EB_no_none[ak.num(photon_EB_no_none,axis=1) > 1]
        EE_photon_selection = photon_EE_no_none[ak.num(photon_EE_no_none,axis=1) > 1]

        EB_photon_selection = EB_photon_selection[:,:2]
        EE_photon_selection = EE_photon_selection[:,:2]
        #pt
        h_photon_EB_pt.fill(dataset_photon_EB_pt=dataset, x=ak.flatten(EB_photon_selection.pt))
        h_photon_EE_pt.fill(dataset_photon_EE_pt=dataset, x=ak.flatten(EE_photon_selection.pt))
        results["photon_EB_pt"] = h_photon_EB_pt
        results["photon_EE_pt"] = h_photon_EE_pt

        #eta
        h_photon_EB_eta.fill(dataset_photon_EB_eta=dataset, x=ak.flatten(EB_photon_selection.eta))
        h_photon_EE_eta.fill(dataset_photon_EE_eta=dataset, x=ak.flatten(EE_photon_selection.eta))

        results["photon_EB_eta"] = h_photon_EB_eta
        results["photon_EE_eta"] = h_photon_EE_eta
        # results["photon_EB_pt"] = photon_EB_selection
        # results["photon_EE_pt"] = photon_EE_selection

        return results
    
    def postprocess(self, accumulator):
        pass


# %%
# samplejson = "/eos/home-h/hhsieh/Run3/mc/2022/GGspin0/0p014/"
for f in os.listdir("."):
    if f.endswith('.json'):
        with open(os.path.join(".", f)) as file:
            sample_dict = json.load(file)

# %%
run = processor.Runner(
    # executor=processor.IterativeExecutor(),
    executor=processor.FuturesExecutor(workers=10), # user 4 cores
    schema=NanoAODSchema
)

results = run(
    sample_dict,
    treename="Events",
    processor_instance=MyProcessor(),
)

# %%
# cell 23
shutil.rmtree('./parquet', ignore_errors=True)
# import mplhep as hep
# import matplotlib.pyplot as plt
plots_dir = './parquet'
Path(plots_dir).mkdir(exist_ok=True)
for key in sample_dict :
    #pT
    value_EB_pt = results[f"photon_EB_pt"][{f"dataset_photon_EB_pt":key}].values()
    value_EE_pt = results[f"photon_EE_pt"][{f"dataset_photon_EE_pt":key}].values()
    value_EB_eta = results[f"photon_EB_eta"][{f"dataset_photon_EB_eta":key}].values()
    value_EE_eta = results[f"photon_EE_eta"][{f"dataset_photon_EE_eta":key}].values()

    parquet_data={
            'pt_EB':value_EB_pt,
            'pt_EE':value_EE_pt,
            'eta_EB':value_EB_eta,
            'eta_EE':value_EE_eta
    }

    df_pt = pd.DataFrame(data=parquet_data)
    df_pt.to_parquet(f'./parquet/{key}_photon.parquet')