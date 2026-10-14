# Attribution

The realistic scene set is drawn with the assets listed here. None of them is part of this repository:
`python -m sgi.synth.realistic.assets fetch` downloads each from its source and checks it against the size
and the SHA-256 that `assets/realistic/manifest/` records. This file is written from that manifest by
`python -m sgi.synth.realistic.assets attribution`; change the manifest, not this file.

Every asset is used in a changed form. A model is scaled to metres, turned to face one way and set on the
ground by the middle of its base; its textures are shrunk, a dense mesh is thinned, and its glass is given
another material. A texture set is projected onto surfaces at its real size, and some are tinted. An avatar
is posed and baked to a still mesh with smaller textures. What was done to one asset alone is noted with it.

## Models

| Asset | Title | Author | Licence | Source | Notes |
|---|---|---|---|---|---|
| `plant_tall` | Potted Plant 01 | Rico Cilliers | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/potted_plant_01 |  |
| `plant_low` | Potted Plant 02 | Rico Cilliers | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/potted_plant_02 |  |
| `bench_station` | Japan Train Station Bench | ditovirnantio (https://sketchfab.com/ditovirnantio) | CC BY 4.0 (the file's own metadata: "CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)"; listed as "by" in Objaverse 1.0, https://huggingface.co/datasets/allenai/objaverse) | https://sketchfab.com/3d-models/japan-train-station-bench-e23d79c1014a431bb913d15fc395b4f0 |  |
| `bench_ottoman` | Red Hook Meknes Storage Ottoman Bench with Fabric Upholstery - 48 x 17 x 16 Inches, Almond (ABO B07H2HGZLK) | Amazon.com | CC BY 4.0 ("Creative Commons Attribution 4.0 International Public License (CC BY 4.0)", credit Amazon.com for the 3D models; https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | https://amazon-berkeley-objects.s3.amazonaws.com/index.html |  |
| `cooler_water` | Water cooler | tboiston (https://sketchfab.com/tboiston) | CC BY 4.0 (the file's own metadata: "CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)"; listed as "by" in Objaverse 1.0, https://huggingface.co/datasets/allenai/objaverse) | https://sketchfab.com/3d-models/water-cooler-4b88c4c4e94c497ca39f831f374e89fc |  |
| `lockers_blue` | Locker | andree (https://sketchfab.com/maestronoov) | CC BY 4.0 (the file's own metadata: "CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)"; listed as "by" in Objaverse 1.0, https://huggingface.co/datasets/allenai/objaverse) | https://sketchfab.com/3d-models/locker-6ab96c76558f43a3b0e1442bc0065f68 |  |
| `chair_task_black` | AmazonBasics Office Chair (ABO B016ID34BO) | Amazon.com | CC BY 4.0 ("Creative Commons Attribution 4.0 International Public License (CC BY 4.0)", credit Amazon.com for the 3D models; https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | https://amazon-berkeley-objects.s3.amazonaws.com/index.html |  |
| `chair_task_grey` | Amazon Brand – Rivet Contemporary Office Desk Chair, 33"-36", Grey/Chrome (ABO B07B7DL3H8) | Amazon.com | CC BY 4.0 ("Creative Commons Attribution 4.0 International Public License (CC BY 4.0)", credit Amazon.com for the 3D models; https://amazon-berkeley-objects.s3.amazonaws.com/index.html) | https://amazon-berkeley-objects.s3.amazonaws.com/index.html |  |
| `monitor_flat` | Monitor | aivarsted (https://sketchfab.com/aivarsted) | CC BY 4.0 (the file's own metadata: "CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)"; listed as "by" in Objaverse 1.0, https://huggingface.co/datasets/allenai/objaverse) | https://sketchfab.com/3d-models/monitor-47558534407a45169544dfb5d86b8c08 |  |
| `keyboard_black` | Keyboards | 3D Share (https://sketchfab.com/3dsharesg) | CC BY 4.0 (the file's own metadata: "CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)"; listed as "by" in Objaverse 1.0, https://huggingface.co/datasets/allenai/objaverse) | https://sketchfab.com/3d-models/keyboards-a25c537932e84ef693de2cdbfc9dd22d |  |
| `coffee_brewer` | Coffee Cart 01 | Joe Seabuhr | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/CoffeeCart_01 | the brewer and its pots taken off the cart they stand on; the cart, its cord and its other things left out |
| `cup_paper` | Paper Coffee Cup | jashmcirwin (https://sketchfab.com/jashmcirwin) | CC BY 4.0 (the file's own metadata: "CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)"; listed as "by" in Objaverse 1.0, https://huggingface.co/datasets/allenai/objaverse) | https://sketchfab.com/3d-models/paper-coffee-cup-6b3afe782b204dfba87a03ddaf222958 | its red wall painted white, with a band of brown round it for a sleeve |
| `bottle_metal` | Water Bottle (glTF Sample Assets) | Microsoft | CC0 1.0 (the model's LICENSE.md: "All files directly associated with the model ... Creative Commons Zero v1.0 Universal"; "This license excludes logos and associated trademarks") | https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main/Models/WaterBottle |  |
| `books_row` | Book Encyclopedia Set 01 | John Malcolm | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/book_encyclopedia_set_01 |  |
| `box_cardboard` | Cardboard Box 01 | Rahul Chaudhary | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/cardboard_box_01 |  |
| `crate_plastic` | Plastic Crate 02 | Fabi_G | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/plastic_crate_02 |  |
| `clock_wall` | Wall Clock | PierreB3D | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/wall_clock |  |
| `suitcase_cabin` | Wahana's Luggage | Naufal Rafi'uddin (https://sketchfab.com/naufalrafiuddin) | CC BY 4.0 (the file's own metadata: "CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)"; listed as "by" in Objaverse 1.0, https://huggingface.co/datasets/allenai/objaverse) | https://sketchfab.com/3d-models/wahanas-luggage-bb7e31d730814e97b882bee710535b12 | the label on its front, which names a travel firm, painted over in the shell's colour |
| `backpack_dark` | Backpack | typhomnt (https://sketchfab.com/typhomnt) | CC BY 4.0 (the file's own metadata: "CC-BY-4.0 (http://creativecommons.org/licenses/by/4.0/)"; listed as "by" in Objaverse 1.0, https://huggingface.co/datasets/allenai/objaverse) | https://sketchfab.com/3d-models/backpack-5655596f9efe47deb09df57a79b44e04 |  |

## Materials

| Asset | Title | Author | Licence | Source | Notes |
|---|---|---|---|---|---|
| `floor_tile` | Interior Tiles | Charlotte Baglioni | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/interior_tiles |  |
| `floor_terrazzo` | Terrazzo Tiles | Amal Kumar | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/terrazzo_tiles | recoloured from its red-brown to a pale warm grey, its chips kept, and its roughness scaled by 0.6 (to about 0.17, smoother than the photograph) so that it shows a soft blurred sheen of the ceiling panels and no mirror image |
| `floor_laminate` | Laminate Floor 02 | Dario Barresi, Charlotte Baglioni | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/laminate_floor_02 |  |
| `ceiling_tile` | OfficeCeiling001 | ambientCG (Lennart Demes) | CC0 1.0 ("licensed under the Creative Commons CC0 1.0 Universal License", https://docs.ambientcg.com/license/) | https://ambientcg.com/a/OfficeCeiling001 |  |
| `wall_paint` | Beige Wall 001 | Dimitrios Savva, Rico Cilliers | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/beige_wall_001 |  |
| `wood` | Oak Veneer 01 | Jenelle van Heerden | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/oak_veneer_01 |  |
| `wood_dark` | Wood Table 001 | Dimitrios Savva, Rico Cilliers | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/wood_table_001 |  |
| `laminate_white` | laminate white, made in code | this project | none: made in code | made in code |  |
| `steel` | Metal009 | ambientCG (Lennart Demes) | CC0 1.0 ("licensed under the Creative Commons CC0 1.0 Universal License", https://docs.ambientcg.com/license/) | https://ambientcg.com/a/Metal009 |  |
| `painted_metal` | painted metal, made in code | this project | none: made in code | made in code |  |
| `plastic_dark` | plastic dark, made in code | this project | none: made in code | made in code |  |
| `plastic_light` | plastic light, made in code | this project | none: made in code | made in code |  |
| `fabric` | Poly Wool Herringbone | colormass, Rico Cilliers | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/poly_wool_herringbone |  |
| `stone` | Granite Tile | Charlotte Baglioni | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/granite_tile |  |
| `cork` | Cork004 | ambientCG (Lennart Demes) | CC0 1.0 ("licensed under the Creative Commons CC0 1.0 Universal License", https://docs.ambientcg.com/license/) | https://ambientcg.com/a/Cork004 |  |
| `paper` | paper, made in code | this project | none: made in code | made in code |  |
| `cardboard` | Cardboard004 | ambientCG (Lennart Demes) | CC0 1.0 ("licensed under the Creative Commons CC0 1.0 Universal License", https://docs.ambientcg.com/license/) | https://ambientcg.com/a/Cardboard004 |  |
| `ceramic` | ceramic, made in code | this project | none: made in code | made in code |  |
| `glass` | glass, made in code | this project | none: made in code | made in code |  |
| `screen` | screen, made in code | this project | none: made in code | made in code |  |

## HDRIs

| Asset | Title | Author | Licence | Source | Notes |
|---|---|---|---|---|---|
| `street_midday` | Wide Street 01 | Sergej Majboroda | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/wide_street_01 |  |
| `sky_partly_cloudy` | Kloofendal 48d Partly Cloudy (Pure Sky) | Greg Zaal, Jarod Guest | CC0 1.0 ("Our assets are all licensed as CC0", https://polyhaven.com/license) | https://polyhaven.com/a/kloofendal_48d_partly_cloudy_puresky |  |

## Fonts

| Asset | Title | Author | Licence | Source | Notes |
|---|---|---|---|---|---|
| `sans` | Lato Regular | Copyright (c) 2010-2014 by tyPoland Lukasz Dziedzic (team@latofonts.com) with Reserved Font Name "Lato" | OFL 1.1 (SIL Open Font License, Version 1.1; the OFL.txt fetched with the font) | https://github.com/google/fonts/tree/main/ofl/lato |  |
| `sans_bold` | Lato Bold | Copyright (c) 2010-2014 by tyPoland Lukasz Dziedzic (team@latofonts.com) with Reserved Font Name "Lato" | OFL 1.1 (SIL Open Font License, Version 1.1; the OFL.txt fetched with the font) | https://github.com/google/fonts/tree/main/ofl/lato |  |

## Avatars

| Asset | Title | Author | Licence | Source | Notes |
|---|---|---|---|---|---|
| `Female_Adult_01` | Female_Adult_01 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_01/Export/Female_Adult_01.fbx |  |
| `Female_Adult_02` | Female_Adult_02 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_02/Export/Female_Adult_02.fbx |  |
| `Female_Adult_03_red` | Female_Adult_03_red | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_03/Export/Female_Adult_03.fbx |  |
| `Female_Adult_04` | Female_Adult_04 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_04/Export/Female_Adult_04.fbx |  |
| `Female_Adult_09` | Female_Adult_09 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_09/Export/Female_Adult_09.fbx |  |
| `Female_Adult_11` | Female_Adult_11 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_11/Export/Female_Adult_11.fbx |  |
| `Female_Adult_12` | Female_Adult_12 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_12/Export/Female_Adult_12.fbx |  |
| `Female_Adult_13` | Female_Adult_13 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_13/Export/Female_Adult_13.fbx |  |
| `Female_Adult_14` | Female_Adult_14 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_14/Export/Female_Adult_14.fbx |  |
| `Female_Adult_15` | Female_Adult_15 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_15/Export/Female_Adult_15.fbx |  |
| `Female_Adult_17_red` | Female_Adult_17_red | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Female_Adult_17/Export/Female_Adult_17.fbx |  |
| `Business_Female_01` | Business_Female_01 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Professions/Business_Female_01/Export/Business_Female_01.fbx |  |
| `Business_Female_03` | Business_Female_03 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Professions/Business_Female_03/Export/Business_Female_03.fbx |  |
| `Chef_Female_01` | Chef_Female_01 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Professions/Chef_Female_01/Export/Chef_Female_01.fbx |  |
| `Pilot_Female_02` | Pilot_Female_02 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Professions/Pilot_Female_02/Export/Pilot_Female_02.fbx |  |
| `Male_Adult_03` | Male_Adult_03 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_03/Export/Male_Adult_03.fbx |  |
| `Male_Adult_04` | Male_Adult_04 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_04/Export/Male_Adult_04.fbx |  |
| `Male_Adult_05` | Male_Adult_05 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_05/Export/Male_Adult_05.fbx |  |
| `Male_Adult_06` | Male_Adult_06 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_06/Export/Male_Adult_06.fbx |  |
| `Male_Adult_09` | Male_Adult_09 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_09/Export/Male_Adult_09.fbx |  |
| `Male_Adult_10` | Male_Adult_10 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_10/Export/Male_Adult_10.fbx |  |
| `Male_Adult_11` | Male_Adult_11 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_11/Export/Male_Adult_11.fbx |  |
| `Male_Adult_12` | Male_Adult_12 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_12/Export/Male_Adult_12.fbx |  |
| `Male_Adult_13` | Male_Adult_13 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_13/Export/Male_Adult_13.fbx |  |
| `Male_Adult_16` | Male_Adult_16 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_16/Export/Male_Adult_16.fbx |  |
| `Male_Adult_17` | Male_Adult_17 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Adults/Male_Adult_17/Export/Male_Adult_17.fbx |  |
| `Business_Male_02` | Business_Male_02 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Professions/Business_Male_02/Export/Business_Male_02.fbx |  |
| `Business_Male_04` | Business_Male_04 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Professions/Business_Male_04/Export/Business_Male_04.fbx |  |
| `Business_Male_06` | Business_Male_06 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Professions/Business_Male_06/Export/Business_Male_06.fbx |  |
| `Security_Male_01` | Security_Male_01 | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Avatars/Professions/Security_Male_01/Export/Security_Male_01.fbx |  |

## Poses

| Asset | Title | Author | Licence | Source | Notes |
|---|---|---|---|---|---|
| `relaxed_a` | relaxed_a | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_idle_neutral_01.max.fbx |  |
| `relaxed_b` | relaxed_b | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_gestic_listen_angry_01.max.fbx |  |
| `relaxed_c` | relaxed_c | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_idle_neutral_03.max.fbx |  |
| `weight_one_leg_a` | weight_one_leg_a | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_idle_neutral_04.max.fbx |  |
| `weight_one_leg_b` | weight_one_leg_b | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_idle_angry_01.max.fbx |  |
| `head_turned_left` | head_turned_left | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_idle_look_around_01.max.fbx |  |
| `head_turned_right` | head_turned_right | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_hold_bag_idle.max.fbx |  |
| `head_turned_right_slight` | head_turned_right_slight | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_idle_neutral_06.max.fbx |  |
| `look_over_shoulder` | look_over_shoulder | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_idle_waiting_01.max.fbx |  |
| `check_watch` | check_watch | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_idle_waiting_02.max.fbx |  |
| `thoughtful_a` | thoughtful_a | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_gestic_thoughtful_01.max.fbx |  |
| `thoughtful_b` | thoughtful_b | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_gestic_thoughtful_01.max.fbx |  |
| `phone_look_a` | phone_look_a | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_cell_phone_textmessage.max.fbx |  |
| `phone_look_b` | phone_look_b | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_cell_phone_textmessage.max.fbx |  |
| `phone_ear` | phone_ear | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_cell_phone_talk_01.max.fbx |  |
| `cup_a` | cup_a | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_drink_idle.max.fbx |  |
| `cup_b` | cup_b | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_drink_idle.max.fbx |  |
| `hand_on_hip` | hand_on_hip | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_gestic_listen_self-assured_01.max.fbx |  |
| `talk_hand_on_chest` | talk_hand_on_chest | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_gestic_talk_cool.max.fbx |  |
| `talk_hand_low` | talk_hand_low | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_gestic_talk_cool.max.fbx |  |
| `talk_lean_in` | talk_lean_in | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_gestic_talk_cool.max.fbx |  |
| `talk_hand_forward` | talk_hand_forward | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_gestic_talk_relaxed_01.max.fbx |  |
| `talk_thumb_back` | talk_thumb_back | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_gestic_talk_relaxed_01.max.fbx |  |
| `talk_open_palm` | talk_open_palm | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_gestic_talk_relaxed_01.max.fbx |  |
| `talk_hand_raised` | talk_hand_raised | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_gestic_talk_relaxed_01.max.fbx |  |
| `talk_both_hands` | talk_both_hands | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_gestic_listen_relaxed_01.max.fbx |  |
| `listen_nod` | listen_nod | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_gestic_listen_neutral_01.max.fbx |  |
| `listen_upright` | listen_upright | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_gestic_listen_self-assured_01.max.fbx |  |
| `serve_upright` | serve_upright | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/m_gestic_listen_neutral_01.max.fbx |  |
| `serve_hands_folded` | serve_hands_folded | Microsoft (Rocketbox Avatar Library) | MIT (LICENSE.md at the repository root, Copyright (c) 2020 Microsoft) | https://raw.githubusercontent.com/microsoft/Microsoft-Rocketbox/master/Assets/Animations/all_animations_max_motextr_static/f_gestic_thoughtful_01.max.fbx |  |

## Licences

- CC0 1.0: https://creativecommons.org/publicdomain/zero/1.0/legalcode
- CC BY 4.0: https://creativecommons.org/licenses/by/4.0/legalcode
- MIT: the text is in the source's own repository, named with each asset
- OFL 1.1: https://openfontlicense.org/open-font-license-official-text/
