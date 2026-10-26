# DavidSpecularity [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23673022.svg)](https://doi.org/10.5281/zenodo.23673022)
Code used to process radar data and create figures in:
>Katz, Z., Ju, H., Young, D., Lee, J., Hills, B., Kang, S., & Siegfried, M. "Mapping David Glacier Subglacial Water Routing with Airborne Ice-Penetrating Radar Doppler Spectra Width," currently in preperation.

## Quickstart
If you need the processed Doppler width maps, they can be found directly at [Zenodo](https://doi.org/10.5281/zenodo.22982329). If you need to process the data yourself to recreate figures from the paper or extend the analysis, follow the steps below:
1. Download the required datasets:

   - **Radar and seismic location data**
      - Dataset archived at [Zenodo](https://doi.org/10.5281/zenodo.22982329)
   
  - **Comparison seismic data**
     - Dataset archived at [KPDC](https://dx.doi.org/10.22663/KOPRI-KPDC-00001978)
     - Note: Data used in Figure 3 available in the 3_Processed_PSTM.zip folder as D2SL2122_PSTM_Stack_Time_domain_static_aperture400_YY2.sgy
   
   - **Altimetry-derived lake outlines**
      - Dataset archived at [Zenodo](https://doi.org/10.5281/zenodo.15758711)
      - Note: Subglacial Cheongsuk Lake Outline at output/lake_outlines/evolving_outlines/David_s1.geojson
   
   - **DDInSAR and Multivariate lake outlines**
      - Dataset archived at [KPDC](https://dx.doi.org/doi:10.22663/KOPRI-KPDC-00001177)
      - Note: DDInSAR outline from DDInSAR.zip; Multivariate outline from SLD2_Boundary.zip

   - **Hydropotential**
      - Dataset archived at [Zenodo](https://doi.org/10.5281/zenodo.16243278)
      - Note: File name subglacial_hydropotential_Antarctica.nc

   - **MODIS Mosaic of Antarctica**
      - Dataset archived at [NSIDC](https://nsidc.org/data/nsidc-0593/versions/2)
      - [Direct Download](https://daacdata.apps.nsidc.org/pub/DATASETS/nsidc0593_moa2009_v02/geotiff/moa750_2009_hp1_v02.0.tif.gz)
      - Note: Zip folder and dataset page labeled v2.0 but actual file names v1.1

   - **MEaSUREs Phase-Based Antarctica Ice Velocity Map, v2**
      - Dataset archived at [NSIDC](https://nsidc.org/data/nsidc-0484/versions/2)
      - [Direct Download](https://data.nsidc.earthdatacloud.nasa.gov/nsidc-cumulus-prod-protected/MEASURES/NSIDC-0484/2/1996/01/01/antarctica_ice_velocity_450m_v2.nc)

   - **RINGS/Bedmap3 Grounding Line**
      - Dataset archived at [Norwegian Polar Institute](https://doi.org/10.21334/NPOLAR.2025.D1062D2A)

3. Clone the repository and use uv to sync the included python environment. Conda users can see the requirements in pyproject.toml.
```bash
git clone git@github.com:zsk4/DavidSpecularity.git
cd DavidSpecularity
uv sync
```
3. Make and move all data to a _Data folder in the repository if desired. When you run a figure plotting script, be sure the paths at the beginning match your data locations.
4. Run desired plotting script, labeled by what figures are created by that script.
   
## Citation
If you use code from this repository, please cite both the publication and the code.
If you use data from the linked [Zenodo](https://doi.org/10.5281/zenodo.22982329) repository, please cite both the publication and the data.
Full citations forthcoming with publication.

