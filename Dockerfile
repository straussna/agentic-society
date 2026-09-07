# Sandbox for one metered episode. Nothing in here is prompt surface:
# the agent never sees this file, only the environment it produces.
FROM debian:bookworm-slim

# bash is required by the prompt's claim; the rest is the toolbox an ordinary
# Debian box has, baked in because the container has no network. Network
# clients are present so their failure is discovered by trying, not inferred
# from a missing binary. Where the model knows two names, both are installed
# (ifconfig/ip, netstat/ss, python/python3). Man pages are reinstated.
RUN rm -f /etc/dpkg/dpkg.cfg.d/docker \
 && apt-get update \
 && apt-get install -y --no-install-recommends --reinstall \
      bash coreutils findutils grep ripgrep fd-find sed gawk diffutils patch moreutils \
      procps psmisc lsof strace inotify-tools util-linux bsdextrautils \
      attr acl libcap2-bin e2fsprogs binutils xxd file less \
      nano vim tree time ncurses-bin ca-certificates man-db manpages \
      python3 python-is-python3 python3-pip python3-setuptools python3-wheel python3-venv \
      git sqlite3 \
      python3-sympy python3-numpy python3-scipy python3-pandas \
      perl openssl jq bc dc zip unzip bzip2 xz-utils zstd uuid-runtime plocate \
      curl wget iproute2 net-tools iputils-ping dnsutils \
      netcat-openbsd socat telnet openssh-client rsync \
 && rm -rf /var/lib/apt/lists/* \
 && ln -s /usr/bin/fdfind /usr/local/bin/fd \
 && updatedb

# Non-root. The agent's trees are copied in and owned by this user, never
# bind-mounted; harness.py probes each for writability before spending anything.
# /work is made here and kept root's - WORKDIR would otherwise create it under
# whatever USER is in force, which is the agent. Every directory below it is made
# when an episode is built, from the channel table in force, so nothing here
# presumes a path and a table naming none of the defaults leaves nothing stale.
# rm and mv ask the directory, not the file, so a harness file is read-only
# because it sits somewhere the agent cannot write.
# The home directory is root's for the same reason: the only place an episode
# can leave anything is a channel, and a channel is what comes back. /tmp stays
# writable because the shell spills heredocs into it, and what an episode leaves
# there is probed for after the fact and recorded as discarded, never lost in
# silence.
RUN useradd --create-home --uid 1000 --shell /bin/bash agent \
 && chown root:root /home/agent && chmod 755 /home/agent \
 && mkdir -p /work && chown root:root /work && chmod 755 /work

USER agent
WORKDIR /work

# The harness drives everything via `docker exec`; the container just needs
# to stay up for the length of one episode.
CMD ["sleep", "infinity"]
