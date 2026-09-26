# DavidSpecularity
Code used to process radar data and create figures in:
>Katz, Z., Ju, H., Young, D., Lee, J., Hills, B., Kang, S., & Siegfried, M. "Mapping David Glacier Subglacial Water Routing with Airborne Ice-Penetrating Radar Doppler Spectra Width," currently in preperation.

## Quickstart
If you need the processed Doppler width maps, they can be found directly at [Zenodo](https://doi.org/10.5281/zenodo.22982329). If you need to process the data yourself to recreate figures from the paper or extend the analysis, follow the steps below:
1. Download the required datasets:
   
   Radar data from [Zenodo](https://doi.org/10.5281/zenodo.22982329)
   
   Comparison seismic data from [KPDC](https://dx.doi.org/10.22663/KOPRI-KPDC-00001978)
   
   Altimetry-derived lake outlines from [Zenodo](https://doi.org/10.5281/zenodo.15758711)
   
   DDInSAR and Multivariate lake outlines from [KPDC](https://dx.doi.org/doi:10.22663/KOPRI-KPDC-00001177)
   
2. Clone the repository and use uv to sync the included python environment. Conda users can see the requirements in pyproject.toml.
```bash
git clone git@github.com:zsk4/DavidSpecularity.git
cd DavidSepcularity
uv sync
```
3. Make and move all data to a _Data folder in the repository if desired. When you run a figure plotting script, be sure the paths at the beginning match your data locations.
4. Run desired plotting script, labeled by what figures are created by that script.
   
## Citation
If you use code from this repository, please cite both the publication and the code.
If you use data from the linked [Zenodo](https://doi.org/10.5281/zenodo.22982329) repository, please cite both the publication and the data.
Full citations forthcoming with publication.
