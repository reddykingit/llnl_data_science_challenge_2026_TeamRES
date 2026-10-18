# Visual review of the marked slices (run on the unified DSC input)

Informational only. No detection was edited. Source: `defect_contact_sheet.png` (one crop per detection), with full marked slices for doubtful ones.

| Group | Detections | Verdict | Reason |
|---|---|---|---|
| Missing struts lying in the slice plane | M001, M002, M006, M009, M010, M021, M022, M025, M026, M027, M032, M033, M039-M042, M055-M058, M067, M073-M075, M079, M083 (26) | confirmed | The red line runs where a strut should be; no material there. M039-M042 and M055-M058 are the two fully removed nodes (12 struts each across slices). |
| Missing struts crossing the slice | remaining 61 of M003-M087 | unclear from one slice | The strut is a dot in this view; the gap is visible as a missing dot in the lattice pattern but needs neighbouring slices to judge. |
| Broken struts lying in the slice plane | B016, B035 | confirmed | Visible break in an in-plane strut. |
| Broken struts lying in the slice plane | B041, B042 | unclear | Mark is very small at this crop size. |
| Broken struts crossing the slice | remaining 70 | unclear from one slice | Small orange gap at a strut cross-section. B001-B003 and B072-B074 sit in the noisy first/last slices. |

Totals: 28 confirmed, 0 marked false alarm by eye, 2 unclear in-plane, 131 cross-section detections not judgeable from a single slice.

For reference (evaluation only, from `score.json`): 9 detections land on struts the reference calls sound (B001, B004, B021, B039, B045, B052, B056, B059, B060), and 2 have the other class (M021 is broken in the reference, B072 is missing).
