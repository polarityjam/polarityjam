# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- selecting feature via categories
- feature for "nucleus_displacement_distance"

### Changed

- increase numpy version
- increase python to 3.10

### Fixed

- Feature of interest normalization via length unit (microns)
- deprecated entry-point loading

## [0.3.0] - 2015-10-06

### Added

- Answer "Should you ever rewrite a change log?".
- symmetry calculation feature "cell_cue_direction_asymmetry"
- plot for cell symmetry

### Changed

- rename "major_to_minor_ratio" to "length_to_width_ratio"

### Fixed

- microsam notebook plotting error

## [0.2.2] - 2024-05-28

### Fixed

- add .npy files to MANIFEST

## [0.2.1] - 2024-04-29

### Added

- readthedocs for build configuration
- plugin structure for CLI entrypoints
- mean and std for circularity statistics

### Changed

- cellpose version 3.0.5

### Fixed

- color for golgi segmentation plot
- cell partition bug when cue_direction != 0
- MACOS compatibility testing

## [0.2.0] - 2023-12-27

### Added

- Zenodo for code citation and long term archiving
- nucleus to cytosol intensity ratio
- statistics for organelle polarity
- color map for marker polarity plot
- cue_direction parameter for orientation plots
- DeepCell segmentation plugin
- cell outlines for marker intensity plot
- configurable channels for segmentation

### Changed

- rename "plot_statistic" to "show_statistic" as plot parameter

# Fixed

- cellpose dimension error for one channel images
- cell-polygon "clockwise" sorting

## [0.1.6] - 2023-09-02

### Added

- circularity as shape index
- documentation for key files

### Changed

- improved import structure
- do not calculate group features per default
- rename "estimated_cell_diameter_nucleus" to "estimated_nucleus_diameter"

### Fixed

- SAM segmentation plugin

## [0.1.5] - 2023-05-12

### Added

- Web-app documentation
- description for Polarity-Index
- description for V-score
- nuclei segmentation and cell centering
- removal of isolated cells
- integration testing for segmentation
- empty mask warning
- test for junction and ratio calculation
- optionally pass segmentation mask of golgi to Polarity-Jam
- 99 percent quantile for cue directional intensity ratio

### Fixed

- contour calculation of cells
- segmentation mask and neighborhood graph synchronization

### Changed

- improved log output
- polarityjam library functionality
- default values for neighborhood graph set to true
- rename "unidirectional" to "axial" for polarity features from 0-180 degrees

## [0.1.4] - 2023-03-03

### Fixed

- fix nuclei displacement plot
- fix neighborhood graph update for deleted cells (nodes)
- fix runtime parameter documentation
- fix "note" box in documentation

## [0.1.3] - 2023-02-22

### Changed

- Improved typing, docstring, and comments in the code.

### Fixed

- fix faulty marker expression plotting

## [0.1.2] - 2023-02-17

### Added

- pypi package for Polarity-Jam.

[unreleased]: https://github.com/polarityjam/polarityjam/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/polarityjam/polarityjam/compare/v0.2.2...v0.3.0
[0.2.2]: https://github.com/polarityjam/polarityjam/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/polarityjam/polarityjam/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/polarityjam/polarityjam/compare/v0.1.6...v0.2.0
[0.1.6]: https://github.com/polarityjam/polarityjam/compare/v0.1.5...v0.1.6
[0.1.5]: https://github.com/polarityjam/polarityjam/compare/v0.1.4...v0.1.5
[0.1.4]: https://github.com/polarityjam/polarityjam/compare/v0.1.3...v0.1.4
[0.1.3]: https://github.com/polarityjam/polarityjam/compare/v0.1.2...v0.1.3
[0.1.2]: https://github.com/polarityjam/polarityjam/releases/tag/v0.1.2
