# shairport-sync

The AirPlay Receiver runs [shairport-sync](https://github.com/mikebrady/shairport-sync) 5.6.3
or newer, built with the pipe backend and metadata support, and finds it on the system `PATH`.
The Docker image builds it for AirPlay 2, but the configuration runs it as classic AirPlay.
`shairport-sync.conf` in this directory is the configuration template for each receiver.

## Docker

The base image (`Dockerfile.base`, `shairport-builder` stage) builds shairport-sync from
source against the image's own FFmpeg. The version and commit are pinned there. It also builds
the `blockaudio` output, whose blocks of audio carry their play times (`block_audio.py` reads
them); that output is on shairport-sync's development branch until 5.7 is released.

## Local development

- **macOS:** `brew install shairport-sync`
- **Linux:** install your distribution's package if it is 5.6.3 or newer. Otherwise build it
  from source with the `./configure` flags of the
  `shairport-builder` stage in `Dockerfile.base`, then `make install`.

`shairport-sync --version` must report 5.6.3 or newer and list `pipe` and `metadata`.
