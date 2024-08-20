Ultramassive atmosphere tables for going from `Teff` and `logg` to photometry (derived from [these models](http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/UMall.html)):
* `CO_Hrich_Massive.csv`
* `CO_Hdef_Massive.csv`
* `ONe_Hrich_Massive.csv`
* `ONe_Hdef_Massive.csv`

Atmosphere tables for all ranges with variable core compositions (derived from [these models](http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/newtables.html)):
* `allwd_Hrich.csv`
* `allwd_Hdef.csv`

*ONe evolutionary tracks* come from [these tables](http://evolgroup.fcaglp.unlp.edu.ar/TRACKS/ultramassive.html), which I use because they are the same ones used by Sihao's WD_models code. They are different from the ultramassive atmosphere tables used to generate the photometry csv's, and they live at:
* `high_mass/ONe_*.dat`

*CO evolutionary tracks* are interpolated directly from the same tables that are used to generate the photometry interpolation table. I think these are the best ultramassive CO tracks available from the La Plata group-- near as I can tell there is no analogy for the above ultramassive ONe evolutionary tracks calculated with a CO core. They can be found at:
* `high_mass/CO_*.dat` 

Everything else is a work in progress that isn't being used yet but will eventually be more detailed atmospheric and evolutionary tracks for He-core WDs.