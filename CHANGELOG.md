# Change Log

## [2.0.0b1] - 2026-08-27

This second release of our 2.0 beta line includes several new features and
improvements, and brings us up to date with main-line development.

All changes from the
[v1.2.0 release notes](https://github.com/NAU-CCL/Epymorph/releases/tag/v1.2.0)
are included in this release.

### Highlights

* :crystal_ball: Improved fitting and forecasting capabilities.
* :bar_chart: New plotting utilities.
* :runner: Movement Models can make use of available mover counts.
* :bug: Bug fixes.

### What's Changed

* Improvements to fitting, including support for multiple strata and
  vector-valued observed data for all filters. In addition,
  `EnsembleKalmanFilterSimulator` now supports all likelihood functions, making
  it suitable for most fitting applications.
* The API for fitting and forecasting has been reorganized. In particular
  several classes have been renamed and/or moved into different modules for
  clarity and consistency.
* A few prototype plotting utilities have been added for multi-realization
  pipeline output. See `PlotRendererPipeline` for the available plots and usage.
* Movement Models can now use available mover counts when computing movement,
  i.e., the total number of individuals currently available to move at each
  location. This is provided as the new `available` argument to
  `MovementClause.evaluate()`.
* _New:_ CDC ADRIO (`InfluenzaStateHospitalizationDaily`) to fetch state-level
  daily influenza data.
* _New:_ Initializer (`RandomLocationsAndRandomSeed`) to randomize both the
  initial infection locations and seed amount.

### Breaking changes

The new movement model functionality is a breaking change, because the added
argument is not optional. Custom movement models implementing `MovementClause`
will need to be updated.

Of lesser consequence for most users, `MovementClause` was previously based on
the `SimulationTickFunction` class, but that class has been removed to simplify
the class hierarchy. `MovementClause` now extends `BaseSimulationFunction`
directly.

## [1.2.0] - 2026-06-30

### Highlights

* :gear: Improved installation issues (as experienced by some users).
* :calendar: Add support for recently updated data sources.
* :bug: Bug fixes.

### Improved installation experience

Some users reported difficulties installing epymorph on some platforms and in
some environments. A significant contributing factor was that we chose to
declare our dependencies with narrow version ranges, in an attempt to ensure
stability and repeatability between environments. However we now realize this
was too restrictive, and just made it challenging to get started with epymorph.
As a result, we've widened our dependency specifications and stepped up efforts
to test and monitor numerical consistency of results across a range of system
profiles and package environments. (We are confident that results are stable
under this dependency scheme, so we have declared this a non-breaking change.)
As always, our goal is to keep epymorph stable to support your scientific
workflows and the repeatability of your work.

### Removing Python 3.13 support declaration

As a consequence of the above, we found we needed to rescind our previous
declaration of support for Python version 3.13. Officially we recommend sticking
with 3.11 or 3.12 for now. We will soon revisit support for Python 3.13 and
3.14, which will require upgrading to numpy v2. While we don't anticipate this
causing real issues, it's a step we want to take carefully!

### What's Changed

* chore: Add support for ACS5 vintage 2024 data.
* chore: Add support for US TIGER vintage 2024 and 2025 geography.
* fix: Centroids movement model numerical overflow when `phi` is small.
  by @Averydx in https://github.com/NAU-CCL/Epymorph/issues/277
* fix: Improved validation when using TimeSelector methods.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/305
* refactor: Compartment models, movement models, and simulation functions
  refactored to not use metaclasses.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/302

## [2.0.0b0] - 2026-02-18

This release kicks off our 2.0 beta release line, which focuses on bringing
enhanced parameter estimation and forecasting tools to epymorph. Vignettes will
follow soon, but for now the main entry points to the new features are in the
`epymorph.forecasting.pipeline` module: classes `ForecastSimulator`,
`ParticleFilterSimulator`, and `EnsembleKalmanFilterSimulator`.

### Beta Note

Because this is a beta release line there may be significant changes between now
and full release, but we wanted to give you access to these features early. We
are currently using this version of epymorph to participate in the CDC FluSight
influenza forecasting challenge so we feel confident in the new features.

### What's Changed

* Added the ability to create multi-realization forecasts from a model.
* Added an ensemble Kalman filter and an improved particle filter for joint
  state and parameter estimation.
  * Support for missing data.
  * Support for more sophisticated relationships between parameters.
* Added the ability to create fitting-to-forecasting pipelines.
* Added ADRIO `CSVFileAxN` which is particularly useful for loading time-series
  data with a non-daily period.

## [1.1.0] - 2025-12-11

This release includes
[caching configuration environment variables](https://github.com/NAU-CCL/Epymorph/issues/275)
(`EPYMORPH_CACHE_DISABLED` and `EPYMORPH_CACHE_DISABLED_PATHS`) and some bug
fixes/updates.

To be on the safe side, we're marking this as a breaking change (hence the minor
version bump) because it does alter API which is publicly exposed. However the
altered API was not directly involved in typical use-cases, so this update will
be effectively non-breaking for most users. (More details below.)

### What's Changed

* Fix bug affecting `LabeledLocations` initializer.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/269
* Fix PRISM ADRIO for data source changes.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/273
* Cache improvements.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/274

### Public API changes

#### Non-breaking

* Added: `epymorph.attribute.NamePattern.conflicts(names)` (static) for checking
  a set of NamePatterns for conflicting/ambiguous names.
* Added: `epymorph.database.Database.query_all(dbs, key)` (static) for finding a
  value in a priority-order list of database instances.
* Changed: in `epymorph.database.ReqTree.of(requirements, params)`, params can
  be given as a list of databases which will be queried in priority order (using
  `Database.query_all`); still compatible with previous value type.

#### Breaking

* Removed: `epymorph.database.DatabaseWithFallback`; fallback behavior is easier
  to handle as a priority list.
* Removed: `epymorph.database.DatabaseWithStrataFallback`; strata parameters can
  be flattened by setting the strata name on a `ModuleNamePattern`.
* Removed: `epymorph.rume.GEO_LABELS`; this was not really used to any effect
  and its existence is potentially confusing.

## [1.0.1] - 2025-09-30

This release brings a few minor changes and fixes focused, mainly, on improved
utility for batch computing environments.

### What's Changed

* MkDocs configuration changes for a multi-version doc site.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/265
* Support IPMs with compartments that have no edges but are included in rate
  equations
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/263
* map and line plot output tools can output to file.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/260
* `sim_messaging()` "liveness" improvements.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/262

## [1.0.0] - 2025-06-30

### First stable release! :tada: 

Having spent the last couple of months testing, updating, polishing, and
documenting we now feel epymorph is sufficiently stable to leave the beta phase.
Changes include:

* The `inspect` functionality added to some ADRIOs in v1.0.0b1 has now been
  extended to every built-in ADRIO.
* More CDC datasets available as ADRIOs.
* The API reference documentation has been revamped and now covers nearly every
  module, class, and function.
* Numerous minor bug fixes and API improvements.

As always, [we'd love to hear](https://docs.www.epimorph.org/about.html) how you
are integrating epymorph in your modeling pipelines!

### What's Changed

* Remove ability to override a RUME's time frame.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/231
* Respiratory Hospitalizations CDC
  by @meaghan66 in https://github.com/NAU-CCL/Epymorph/pull/235
* LODES and PRISM estimate fix.
  by @meaghan66 in https://github.com/NAU-CCL/Epymorph/pull/236
* Improved how anomalous floating-point calculations are handled in the particle
  filter resamplers.
  by @JeffreyCovington in https://github.com/NAU-CCL/Epymorph/pull/238
* Fix DateRange interface problem.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/240
* Error message improvement for invalid selections.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/234
* Fix error message when there are parameter evaluation issues.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/243
* Add DataResolver.get_raw method.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/245
* ADRIO refactor
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/246
* ADRIO minor improvements.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/248
* Switch to MkDocs for API documentation site generation.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/251
* Add abstract `geography` property to `GeoScope`.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/252
* ADRIO v2 conversions
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/253
* API docs and small improvements
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/254

## [1.0.0b1] - 2025-04-25

### _You beta believe it_, it's an improved beta version!

This release focuses on improving workflows using ADRIOs. The real world is
messy and full of caveats, so we introduced standardized methods for addressing
data issues such as redacted or missing data. In addition, we added the
[inspect](https://docs.www.epimorph.org/api/InspectResult.html#examples) method
to help discover and resolve issues.

So far only a selected few ADRIOs have been converted to the "new style"
interface (`acs5`, `cdc`, and `commuting_flows`) but we plan to convert the rest
of them soon. We believe this is a major improvement in data integrity and ease
of use.

Meanwhile we continue to improve our API documentation and squash minor bugs.
Enjoy!

### What's Changed

* LODES and PRISM estimate fix.
  by @meaghan66 in https://github.com/NAU-CCL/Epymorph/pull/236
* Improved how anomalous floating-point calculations are handled in the particle
  filter resamplers.
  by @JeffreyCovington in https://github.com/NAU-CCL/Epymorph/pull/238
* Fix DateRange interface problem.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/240
* Error message improvement for invalid selections.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/234
* Fix error message when there are parameter evaluation issues.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/243
* Add DataResolver.get_raw method.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/245
* ADRIO refactor
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/246
* ADRIO minor improvements.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/248

## [1.0.0b0] - 2025-02-13

:tada: epymorph is now in v1.0.0 beta! :tada:

The main features of epymorph are in place and we believe it's ready for
enthusiasts outside of our team to try it out. Your feedback is critical and
appreciated, especially during this beta period. We expect there may be some bug
fixes and quality-of-life improvements between now and removing the beta tag,
but no major feature additions, removals, or breaking refactors. Enjoy!

### What's Changed

* Remove ability to override a RUME's time frame.
  by @JavadocMD in https://github.com/NAU-CCL/Epymorph/pull/231
* Respiratory Hospitalizations CDC
  by @meaghan66 in https://github.com/NAU-CCL/Epymorph/pull/235

[2.0.0b0]: https://github.com/NAU-CCL/Epymorph/compare/v1.2.0...v2.0.0b1
[1.2.0]: https://github.com/NAU-CCL/Epymorph/compare/v1.1.0...v1.2.0
[2.0.0b1]: https://github.com/NAU-CCL/Epymorph/compare/v1.1.0...v2.0.0b0
[1.1.0]: https://github.com/NAU-CCL/Epymorph/compare/v1.0.1...v1.1.0
[1.0.1]: https://github.com/NAU-CCL/Epymorph/compare/v1.0.0...v1.0.1
[1.0.0]: https://github.com/NAU-CCL/Epymorph/compare/v0.10.0...v1.0.0
[1.0.0b1]: https://github.com/NAU-CCL/Epymorph/compare/v1.0.0b0...v1.0.0b1
[1.0.0b0]: https://github.com/NAU-CCL/Epymorph/compare/v0.10.0...v1.0.0b0
