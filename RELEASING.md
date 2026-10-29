# Releasing AppleNeuralEngine.jl

Julia packages are released by **registering** a version in the
[General registry](https://github.com/JuliaRegistries/General). After that,
`Pkg.add("AppleNeuralEngine")` installs it. The version number comes from
`Project.toml`; the registry and TagBot create the git tag and GitHub release.

## Version numbers

The package uses [semantic versioning](https://semver.org/) as Julia's package
manager interprets it:

| change | before 1.0 | from 1.0 |
|---|---|---|
| breaking change (removed/renamed API, changed behavior) | 0.**x** → 0.**x+1** | **x** → **x+1** |
| new feature, backward compatible | 0.x.**y** → 0.x.**y+1** | x.**y** → x.**y+1** |
| bug fix only | 0.x.**y** → 0.x.**y+1** | x.y.**z** → x.y.**z+1** |

Before 1.0 the minor number acts as the "major" one: users' `[compat]` entry
`AppleNeuralEngine = "0.1"` accepts 0.1.x but not 0.2. Release 1.0 once the API
has settled.

## Checklist for a release

1. Tests pass locally on Apple Silicon, *with* the Neural Engine:
   `julia --project=. -e 'using Pkg; Pkg.test()'`.
2. Examples run: `julia --project=. examples/minimal.jl`,
   `julia --project=. examples/transformer_ffn.jl`.
3. Optionally refresh the op table: `julia --project=. tools/probe_ops.jl`.
4. Docs build: `julia --project=docs docs/make.jl`.
5. Set `version` in `Project.toml` and move the `[Unreleased]` notes in
   `CHANGELOG.md` under the new version with today's date.
6. Commit to `main`, push, wait for CI to pass.
7. Register: comment on the commit (or an issue) on GitHub

       @JuliaRegistrator register

   with release notes in the comment if you like. The Registrator bot opens a pull
   request on General. Automatic checks run, and a *new* package waits 3 days
   before merging. Later versions merge in about 15 minutes.
8. When it merges, TagBot creates the tag `v0.x.y` and a GitHub release, and the
   documentation for that version is published.

## One-time setup (first release only)

- Create the GitHub repository `AppleNeuralEngine.jl` (the name must end in `.jl`)
  and push `main`.
- Install the [JuliaRegistrator GitHub app](https://github.com/JuliaRegistries/Registrator.jl)
  on the repository.
- Documentation deployment: run `using DocumenterTools; DocumenterTools.genkeys(user="pitsianis", repo="AppleNeuralEngine.jl")`,
  add the public key as a deploy key (with write access) and the private key as
  the repository secret `DOCUMENTER_KEY`. Then enable GitHub Pages on the
  `gh-pages` branch.
