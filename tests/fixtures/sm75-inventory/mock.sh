#!/bin/bash
set -uo pipefail
tool=${0##*/}
printf '%s' "$tool" >> "$MOCK_LOG"
printf ' <%s>' "$@" >> "$MOCK_LOG"
printf '\n' >> "$MOCK_LOG"
case "$tool" in
    sha256sum)
        fingerprint=${MOCK_FINGERPRINT:-5c46e8b753855406abd9880d52d6d9361c290f264c0baaa88255c8680aa42414}
        if [[ ${1:-} == */.executable-copy.partial ]]; then fingerprint=${MOCK_COPY_FINGERPRINT:-$fingerprint}; fi
        printf '%s  %s\n' "$fingerprint" "${1:-}"
        ;;
    wc)
        if [[ -n ${MOCK_BINARY_SIZE:-} ]]; then
            printf '%s %s\n' "$MOCK_BINARY_SIZE" "${2:-}"
        else
            /usr/bin/wc "$@"
        fi
        ;;
    readelf)
        printf 'ELF metadata only: %s\n' "$*"
        ;;
    cuobjdump|nvcc)
        printf '%s metadata only: %s\n' "$tool" "$*"
        ;;
    nsys)
        if [[ ${MOCK_SLEEP_HELP:-0} == 1 && ${1:-} == profile ]]; then sleep 3; fi
        printf 'nsys metadata only: %s\n' "$*"
        ;;
    dpkg-query)
        case ${1:-} in
            --search) printf 'cuda-test:amd64: %s\n' "${3:-}" ;;
            --show) printf 'cuda-test:amd64\t13.2-test\tamd64\tinstall ok installed\n' ;;
            *) exit 91 ;;
        esac
        ;;
    sudo)
        if [[ ${1:-} == -v ]]; then exit "${MOCK_SUDO_STATUS:-0}"; fi
        if [[ ${1:-} != -n || ${2:-} != -- ]]; then exit 92; fi
        shift 2
        "$@"
        ;;
    dmidecode)
        if [[ ${1:-} != --type || ! ${2:-} =~ ^(0|1|2|39)$ ]]; then exit 93; fi
        printf 'DMI type %s read-only fixture\n' "$2"
        ;;
    ipmitool)
        case "$*" in 'sel elist'|'sdr type Power Supply'|'sensor list'|'fru print') ;;
            *) exit 94 ;;
        esac
        if [[ ${MOCK_SLEEP_BMC:-0} == 1 && "$*" == 'sel elist' ]]; then sleep 3; fi
        printf 'BMC read-only fixture: %s\n' "$*"
        ;;
    *) printf 'unexpected tool: %s\n' "$tool" >&2; exit 95 ;;
esac
