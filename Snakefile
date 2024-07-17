#########################################################################################################
# MAIN
#########################################################################################################
configfile: "snakemake-config.yaml"

rule megsai_all:
    input:
        checkpoint = expand(config["model"]["checkpoint_path"]+"/{instrument}/"+config["model"]["checkpoint_file"]+".ckpt", instrument=config["model"]["instruments"]),
        results_train = expand(config["model"]["checkpoint_path"]+"/{instrument}/train_irradiance.npy", instrument=config["model"]["instruments"]),
        results_valid = expand(config["model"]["checkpoint_path"]+"/{instrument}/valid_irradiance.npy", instrument=config["model"]["instruments"]),
        results_test = expand(config["model"]["checkpoint_path"]+"/{instrument}/test_irradiance.npy", instrument=config["model"]["instruments"]),
        instrument_mape = config["model"]["checkpoint_path"]+"/overview_instruments_mape.png",
        instrument_mepe = config["model"]["checkpoint_path"]+"/overview_instruments_mepe.png",
        instrument_mae = config["model"]["checkpoint_path"]+"/overview_instruments_mae.png",
        instrument_mee = config["model"]["checkpoint_path"]+"/overview_instruments_mee.png",
        wavelength_mape = config["model"]["checkpoint_path"]+"/overview_wavelengths_mape.png",
        wavelength_mepe = config["model"]["checkpoint_path"]+"/overview_wavelengths_mepe.png",
        wavelength_mae = config["model"]["checkpoint_path"]+"/overview_wavelengths_mae.png",
        wavelength_mee = config["model"]["checkpoint_path"]+"/overview_wavelengths_mee.png",
        # frame = config["model"]["checkpoint_path"]+"/overview_frame_AIA_0.png"
        matches_table = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_table"],
        matches_tableb = config["data"]["matches_euvi_path"]+"/"+config["data"]["matches_euvi_table"],
        forecast_table = config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_euvib_table"],
        stereo_table = config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_euvi_table"] #,
        # frame = config["model"]["checkpoint_path"]+"/images/MEGS_AI_AIA_EUVI_134.png",
        # forecast_results = config["model"]["checkpoint_path"]+"/EUVI/images/forecast_aia1_0_FeXVIII.png",
        # forecast_results2 = config["model"]["checkpoint_path"]+"/EUVI/images/forecast_aia2_0_FeXVIII.png"

#########################################################################################################
# SETUP AND GENERATE TRAINING DATA
#########################################################################################################

## Download GOES soft X-ray flux data
rule download_goes_data:
    output: 
        goes_data = config["data"]["goes_path"]+"/"+config["data"]["goes_data"]
    params:
        output_path = config["data"]["goes_path"]
    shell:
        """
        mkdir -p {params.output_path} && 
        gsutil -m cp -r gs://us-4pieuvirradiance-dev-data/goes.csv {output.goes_data}
        """        

## Generate CDF file containing MEGS-A irradiance data
rule generate_eve_netcdf:
    input:
        eve_path = config["data"]["eve_path"]
    output:
        megsa_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_data"]
    params:
        output_path = config["data"]["megsa_path"]
    shell:
        """
        mkdir -p {params.output_path} &&
        python s4pi/irradiance/preprocess/generate_eve_netcdf.py \
        -eve_path {input.eve_path} \
        -output {output.megsa_data}
        """  

## Generates matches between MEGS-A data and AIA data
rule generate_matches_time:
    input:
        goes_data = config["data"]["goes_path"]+"/"+config["data"]["goes_data"],
        megsa_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_data"],
        aia_path = config["data"]["aia_path"]
    output:
        matches_table = config["data"]["matches_path"]+"/"+config["data"]["matches_table"]
    params:
        eve_cutoff = config["data"]["eve_cutoff"],
        aia_cutoff = config["data"]["aia_cutoff"], 
        aia_wl = config["data"]["aia_wl"],
        matches_path = config["data"]["matches_path"],
        debug = config["debug"]
    shell:
        """
        mkdir -p {params.matches_path} &&
        python -m s4pi.irradiance.preprocess.generate_matches_time \
        -goes_path {input.goes_data} \
        -eve_path {input.megsa_data} \
        -aia_path {input.aia_path} \
        -output_path {output.matches_table} \
        -aia_wl {params.aia_wl} \
        -eve_cutoff {params.eve_cutoff} \
        -aia_cutoff {params.aia_cutoff} \
        -debug {params.debug}
        """

## Preprocess MEGS-A data
rule generate_eve_ml_ready:
    input:
        eve_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_data"],
        matches_table = config["data"]["matches_path"]+"/"+config["data"]["matches_table"]
    output:
        eve_converted_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_converted_data"],
        eve_norm = config["data"]["megsa_path"]+"/"+config["data"]["megsa_norm"],
        eve_wl = config["data"]["megsa_path"]+"/"+config["data"]["megsa_wl"]
    shell:
        """
            python -m s4pi.irradiance.preprocess.generate_eve_ml_ready \
            -eve_path {input.eve_data} \
            -matches_table {input.matches_table} \
            -output_data {output.eve_converted_data} \
            -output_norm {output.eve_norm} \
            -output_wl {output.eve_wl}
        """

## Generates donwnscaled stacks of the AIA channels
rule generate_euv_image_stacks:
    input:
        aia_path = config["data"]["aia_path"],
        matches_table = config["data"]["matches_path"]+"/"+config["data"]["matches_table"]
    params:
        aia_resolution = config["data"]["aia_resolution"],
        aia_reproject = config["data"]["aia_reproject"],
        matches_stacks = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_stacks"],
        debug = config["debug"]
    output:
        matches_table = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_table"],
        aia_stats = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["aia_stats"]
    shell:
        """
        mkdir -p {params.matches_stacks} &&
        python -m s4pi.irradiance.preprocess.generate_euv_image_stacks \
        -aia_path {input.aia_path} \
        -aia_resolution {params.aia_resolution} \
        -aia_reproject {params.aia_reproject} \
        -aia_stats {output.aia_stats} \
        -matches_table {input.matches_table} \
        -matches_output {output.matches_table} \
        -matches_stacks {params.matches_stacks} \
        -debug {params.debug}
        """
        

#########################################################################################################
# TRAIN & TEST MODEL
#########################################################################################################

rule megsai_train:
    input:
        matches_table = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_table"],
        eve_converted_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_converted_data"],
        eve_norm = config["data"]["megsa_path"]+"/"+config["data"]["megsa_norm"],
        eve_wl = config["data"]["megsa_path"]+"/"+config["data"]["megsa_wl"]
    params:
        instrument = "{instrument}",
        config_file = config["model"]["config_file"],
        checkpoint_path = config["model"]["checkpoint_path"]+"/{instrument}",
        checkpoint_file = config["model"]["checkpoint_path"]+"/{instrument}/"+config["model"]["checkpoint_file"]
    output: 
        checkpoint = config["model"]["checkpoint_path"]+"/{instrument}/"+config["model"]["checkpoint_file"]+".ckpt"
    resources:
        nvidia_gpu = 1
    shell:
        """
        mkdir -p {params.checkpoint_path} &&
        python -m s4pi.irradiance.train \
        -checkpoint {params.checkpoint_file} \
        -model {params.config_file} \
        -matches_table {input.matches_table} \
        -eve_data {input.eve_converted_data} \
        -eve_norm {input.eve_norm} \
        -eve_wl {input.eve_wl} \
        -instrument {params.instrument}
        """

rule megsai_test:
    input:
        goes_data = config["data"]["goes_path"]+"/"+config["data"]["goes_data"], 
        matches_table = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_table"],
        eve_converted_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_converted_data"],
        eve_norm = config["data"]["megsa_path"]+"/"+config["data"]["megsa_norm"],
        eve_wl = config["data"]["megsa_path"]+"/"+config["data"]["megsa_wl"],
        checkpoint = config["model"]["checkpoint_path"]+"/{instrument}/"+config["model"]["checkpoint_file"]+".ckpt"
    params:
        output_path = config["model"]["checkpoint_path"]+"/{instrument}"
    output: 
        results_train = config["model"]["checkpoint_path"]+"/{instrument}/train_irradiance.npy",
        results_valid = config["model"]["checkpoint_path"]+"/{instrument}/valid_irradiance.npy",
        results_test = config["model"]["checkpoint_path"]+"/{instrument}/test_irradiance.npy"
    resources:
        nvidia_gpu = 1
    shell:
        """
        mkdir -p {params.output_path} &&
        python -m s4pi.irradiance.evaluation.plot_megsa \
        -checkpoint {input.checkpoint} \
        -goes_data {input.goes_data} \
        -matches_table {input.matches_table} \
        -eve_data {input.eve_converted_data} \
        -eve_norm {input.eve_norm} \
        -eve_wl {input.eve_wl} \
        -output_path {params.output_path} 
        """

rule megsai_summary_instruments:
    input:
        matches_table = config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_table"],
        eve_converted_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_converted_data"],
        eve_norm = config["data"]["megsa_path"]+"/"+config["data"]["megsa_norm"],
        eve_wl = config["data"]["megsa_path"]+"/"+config["data"]["megsa_wl"],
        aia_ckpt = config["model"]["checkpoint_path"]+"/AIA/"+config["model"]["checkpoint_file"]+".ckpt",
        euvi_ckpt = config["model"]["checkpoint_path"]+"/EUVI/"+config["model"]["checkpoint_file"]+".ckpt",
        eui_ckpt = config["model"]["checkpoint_path"]+"/EUI/"+config["model"]["checkpoint_file"]+".ckpt",
        suvi_ckpt = config["model"]["checkpoint_path"]+"/SUVI/"+config["model"]["checkpoint_file"]+".ckpt",
        circe_ckpt = config["model"]["checkpoint_path"]+"/CIRCE/"+config["model"]["checkpoint_file"]+".ckpt"
    params:
        checkpoint_path = config["model"]["checkpoint_path"], 
        mission = [config["data"]["aia_mission"], config["data"]["euvi_mission"], config["data"]["eui_mission"], config["data"]["suvi_mission"], config["data"]["circe_mission"]],
        instrument = config["model"]["instruments"][9:],
        instrument_output = config["model"]["checkpoint_path"]+"/overview_instruments", 
        wavelength = config["model"]["instruments"][:9],
        wavelength_output = config["model"]["checkpoint_path"]+"/overview_wavelengths",
        output_path = config["model"]["checkpoint_path"]
    output: 
        instrument_mape = config["model"]["checkpoint_path"]+"/overview_instruments_mape.png",
        instrument_mepe = config["model"]["checkpoint_path"]+"/overview_instruments_mepe.png",
        instrument_mae = config["model"]["checkpoint_path"]+"/overview_instruments_mae.png",
        instrument_mee = config["model"]["checkpoint_path"]+"/overview_instruments_mee.png",
        wavelength_mape = config["model"]["checkpoint_path"]+"/overview_wavelengths_mape.png",
        wavelength_mepe = config["model"]["checkpoint_path"]+"/overview_wavelengths_mepe.png",
        wavelength_mae = config["model"]["checkpoint_path"]+"/overview_wavelengths_mae.png",
        wavelength_mee = config["model"]["checkpoint_path"]+"/overview_wavelengths_mee.png"
    resources:
        nvidia_gpu = 1
    shell:
        """
        mkdir -p {params.output_path} &&
        python -m s4pi.irradiance.evaluation.plot_instruments \
        -checkpoint_path {params.checkpoint_path} \
        -instrument {params.instrument} \
        -mission {params.mission} \
        -eve_data {input.eve_converted_data} \
        -eve_norm {input.eve_norm} \
        -eve_wl {input.eve_wl} \
        -matches_table {input.matches_table} \
        -output_path {params.instrument_output} &&
        python -m s4pi.irradiance.evaluation.plot_instruments \
        -checkpoint_path {params.checkpoint_path} \
        -instrument {params.wavelength} \
        -eve_data {input.eve_converted_data} \
        -eve_norm {input.eve_norm} \
        -eve_wl {input.eve_wl} \
        -matches_table {input.matches_table} \
        -output_path {params.wavelength_output}
        """

## Generates matches between MEGS-A data and AIA data
rule generate_matches_stereo:
    input:
        goes_data = config["data"]["goes_path"]+"/"+config["data"]["goes_data"],
        megsa_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_data"],
        inst_path = [config["data"]["aia_path"], config["data"]["euvia_path"], config["data"]["euvib_path"]]
    output:
        matches_table = config["data"]["matches_euvi_path"]+"/"+config["data"]["matches_euvi_table"]
    params:
        inst = ['AIA', 'A/EUVI', 'B/EUVI'],
        eve_cutoff = config["data"]["eve_cutoff"],
        inst_cutoff = config["data"]["euvi_cutoff"], 
        inst_wl0 = config["data"]["aia_wl"],
        inst_wl1 = config["data"]["euvi_wl"],
        inst_wl2 = config["data"]["euvi_wl"],
        matches_path = config["data"]["matches_euvi_path"]
    shell:
        """
        mkdir -p {params.matches_path} &&
        python -m s4pi.irradiance.preprocess.generate_matches_time_stereo \
        -goes_path {input.goes_data} \
        -eve_path {input.megsa_data} \
        -instrument {params.inst} \
        -instrument_path {input.inst_path} \
        -output_path {output.matches_table} \
        -eve_cutoff {params.eve_cutoff} \
        -instrument_cutoff {params.inst_cutoff}
        """

## Generates matches between MEGS-A data and AIA data
rule generate_matches_stereob:
    input:
        goes_data = config["data"]["goes_path"]+"/"+config["data"]["goes_data"],
        megsa_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_data"],
        inst_path = [config["data"]["euvib_path"]]
    output:
        matches_table = config["data"]["matches_euvi_path"]+"/"+config["data"]["matches_euvib_table"]
    params:
        inst = ['B/EUVI'],
        delay = True,
        eve_cutoff = config["data"]["eve_cutoff"],
        inst_cutoff = config["data"]["euvi_cutoff"], 
        matches_path = config["data"]["matches_euvi_path"]
    shell:
        """
        mkdir -p {params.matches_path} &&
        python -m s4pi.irradiance.preprocess.generate_matches_time_stereo \
        -goes_path {input.goes_data} \
        -eve_path {input.megsa_data} \
        -instrument {params.inst} \
        -instrument_path {input.inst_path} \
        -output_path {output.matches_table} \
        -eve_cutoff {params.eve_cutoff} \
        -instrument_cutoff {params.inst_cutoff} \
        -delay {params.delay}
        """

## Generates donwnscaled stacks of the AIA channels
rule generate_stacks_stereo:
    input:
        aia_path = config["data"]["aia_path"],
        matches_table = config["data"]["matches_euvi_path"]+"/"+config["data"]["matches_euvi_table"]
    params:
        inst = ['AIA', 'A/EUVI', 'B/EUVI'],
        aia_resolution = config["data"]["aia_resolution"],
        aia_reproject = config["data"]["aia_reproject"],
        matches_stacks = [config["data"]["matches_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_stacks"], config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_stacks"], config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_stacks"]],
        debug = config["debug"]
    output:
        matches_table = config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_euvi_table"],
        inst_stats = config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["euvi_stats"]
    shell:
        """
        mkdir -p {params.matches_stacks} &&
        python -m s4pi.irradiance.preprocess.generate_euv_image_stacks_stereo \
        -aia_path {input.aia_path} \
        -aia_resolution {params.aia_resolution} \
        -aia_reproject {params.aia_reproject} \
        -aia_stats {output.inst_stats} \
        -instrument {params.inst} \
        -matches_table {input.matches_table} \
        -matches_output {output.matches_table} \
        -matches_stacks {params.matches_stacks} \
        -debug {params.debug}
        """

## Generates donwnscaled stacks of the AIA channels
rule generate_stacks_stereob:
    input:
        aia_path = config["data"]["aia_path"],
        matches_table = config["data"]["matches_euvi_path"]+"/"+config["data"]["matches_euvib_table"]
    params:
        inst = ['B/EUVI'],
        aia_resolution = config["data"]["aia_resolution"],
        aia_reproject = config["data"]["aia_reproject"],
        matches_stacks = config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_stacks"],
        debug = config["debug"]
    output:
        matches_table = config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_euvib_table"],
        inst_stats= config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["euvi_stats"]
    shell:
        """
        mkdir -p {params.matches_stacks} &&
        python -m s4pi.irradiance.preprocess.generate_euv_image_stacks_stereo \
        -aia_path {input.aia_path} \
        -aia_resolution {params.aia_resolution} \
        -aia_reproject {params.aia_reproject} \
        -aia_stats {output.inst_stats} \
        -instrument {params.inst} \
        -matches_table {input.matches_table} \
        -matches_output {output.matches_table} \
        -matches_stacks {params.matches_stacks} \
        -debug {params.debug}
        """

rule generate_eve_stereo_ready:
    input:
        eve_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_data"],
        matches_table = config["data"]["matches_euvi_path"]+"/"+config["data"]["matches_euvi_table"]
    output:
        eve_converted_data = config["data"]["matches_euvi_path"]+"/"+config["data"]["megsa_converted_data"],
        eve_norm = config["data"]["matches_euvi_path"]+"/"+config["data"]["megsa_norm"],
        eve_wl = config["data"]["matches_euvi_path"]+"/"+config["data"]["megsa_wl"]
    shell:
        """
            python -m s4pi.irradiance.preprocess.generate_eve_ml_ready \
            -eve_path {input.eve_data} \
            -matches_table {input.matches_table} \
            -output_data {output.eve_converted_data} \
            -output_norm {output.eve_norm} \
            -output_wl {output.eve_wl}
        """

rule megsai_stereo_frames:
    input:
        matches_table = config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_euvi_table"],
        eve_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_data"],
        eve_norm = config["data"]["megsa_path"]+"/"+config["data"]["megsa_norm"],
        eve_wl = config["data"]["megsa_path"]+"/"+config["data"]["megsa_wl"],
        aia_ckpt = config["model"]["checkpoint_path"]+"/AIA/"+config["model"]["checkpoint_file"]+".ckpt",
        euvi_ckpt = config["model"]["checkpoint_path"]+"/EUVI/"+config["model"]["checkpoint_file"]+".ckpt"
    params: 
        checkpoint_path = config["model"]["checkpoint_path"],
        output_path = config["model"]["checkpoint_path"]+'/images/MEGS_AI',
        frames = [134, 195]  # frames = [4997, 5487] 
    output: 
        frame = config["model"]["checkpoint_path"]+"/images/MEGS_AI_AIA_EUVI_134.png"
    resources:
        nvidia_gpu = 1
    shell:
        """
        mkdir -p {params.output_path} &&
        python -m s4pi.irradiance.evaluation.plot_frames \
        -checkpoint_path {params.checkpoint_path} \
        -eve_data {input.eve_data} \
        -eve_norm {input.eve_norm} \
        -eve_wl {input.eve_wl} \
        -matches_table {input.matches_table} \
        -output_path {params.output_path} \
        -frames {params.frames}
        """

rule megsai_stereob_forecast:
    input:
        matches_table = config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_euvib_table"],
        eve_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_data"],
        eve_norm = config["data"]["megsa_path"]+"/"+config["data"]["megsa_norm"],
        eve_wl = config["data"]["megsa_path"]+"/"+config["data"]["megsa_wl"],
        checkpoint = config["model"]["checkpoint_path"]+"/EUVI/"+config["model"]["checkpoint_file"]+".ckpt"
    params:
        output_path = config["model"]["checkpoint_path"]+"/EUVI/images"
    output: 
        results = config["model"]["checkpoint_path"]+"/EUVI/images/forecast_aia1_0_FeXVIII.png"
    resources:
        nvidia_gpu = 1
    shell:
        """
        mkdir -p {params.output_path} &&
        python -m s4pi.irradiance.evaluation.plot_forecast \
        -checkpoint {input.checkpoint} \
        -matches_table {input.matches_table} \
        -eve_data {input.eve_data} \
        -eve_norm {input.eve_norm} \
        -eve_wl {input.eve_wl} \
        -output_path {params.output_path} 
        """

rule megsai_stereob_forecast2:
    input:
        matches_table = config["data"]["matches_euvi_path"]+"/"+str(config["data"]["aia_resolution"])+"/"+config["data"]["matches_euvib_table"],
        eve_data = config["data"]["megsa_path"]+"/"+config["data"]["megsa_data"],
        eve_norm = config["data"]["megsa_path"]+"/"+config["data"]["megsa_norm"],
        eve_wl = config["data"]["megsa_path"]+"/"+config["data"]["megsa_wl"],
        checkpoint = config["model"]["checkpoint_path"]+"/EUVI/"+config["model"]["checkpoint_file"]+".ckpt"
    params:
        output_path = config["model"]["checkpoint_path"]+"/EUVI/images"
    output:
        results = config["model"]["checkpoint_path"]+"/EUVI/images/forecast_aia2_0_FeXVIII.png"
    resources:
        nvidia_gpu = 1
    shell:
        """
        mkdir -p {params.output_path} &&
        python -m s4pi.irradiance.evaluation.plot_forecast_hist \
        -checkpoint {input.checkpoint} \
        -matches_table {input.matches_table} \
        -eve_data {input.eve_data} \
        -eve_norm {input.eve_norm} \
        -eve_wl {input.eve_wl} \
        -output_path {params.output_path} 
        """