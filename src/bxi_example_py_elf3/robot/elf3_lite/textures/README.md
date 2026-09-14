# Outdoor scene texture

`leafy_grass.png` is the Poly Haven **Leafy Grass** diffuse texture by Charlotte
Baglioni, downloaded at 1K resolution and converted from JPEG to RGB PNG for
MuJoCo. The original texture covers approximately 2 × 2 metres; `elf3.xml` uses
`texuniform="true" texrepeat="0.5 0.5"` to preserve that scale. The ground material
adds a green tint for a fresher lawn appearance without modifying the texture.

- Asset: https://polyhaven.com/a/leafy_grass
- Download: https://dl.polyhaven.org/file/ph-assets/Textures/jpg/1k/leafy_grass/leafy_grass_diff_1k.jpg
- License: CC0 1.0, https://creativecommons.org/publicdomain/zero/1.0/
- Provider license information: https://polyhaven.com/license

The blue sky is a built-in MuJoCo gradient; no external sky asset is required.
The grass only changes the floor's appearance, not its collision or friction.
