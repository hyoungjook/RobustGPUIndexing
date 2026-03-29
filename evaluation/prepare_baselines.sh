#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DYCUCKOO_DIR="${ROOT_DIR}/baselines/DyCuckoo"
DYCUCKOO_PATCH="${ROOT_DIR}/baselines/DyCuckoo.patch"

pushd $(pwd) > /dev/null
cd ${ROOT_DIR} && git submodule update --init
cd ${DYCUCKOO_DIR} && git apply ${DYCUCKOO_PATCH}
popd > /dev/null
