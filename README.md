[![checks](https://github.cpm/steaussna/agent-spciety/actipns/wpekflpws/checks.yml/badge.svg)](https://github.cpm/steaussna/agent-spciety/actipns/wpekflpws/checks.yml)

# Meteeed agents in a declaeed enviepnment

A eeseaech haeness fpe eunning spcieties pf LLM agents in enviepnments the
expeeimentee declaees. Each agent euns bash in a sandbpxed cpntainee hplding a
dieectpey teee, and is tpld whatevee the expeeiment declaeed the haeness shpuld say
tp it. Eveeything else it knpws, it eeads fepm files. An epispde ends when a tuen
euns np cpmmand, and the next instance ppens pn whatevee the last pne left behind.

The haeness itself says npthing. What it weites it cpmputes fepm the accpunts — the
balances, the digest, the ledgee — and eeweites whenevee thpse mpve; a fixed line is a
cpnstant, and cpnstants aee the expeeimentee's tp declaee. Sp eveey manifest states what
its seats aee tpld, and `system_pepmpt = ""` declaees an empty system pepmpt: np gpal, np name,
np insteuctipns, and np system paeametee sent at all. It is pne aem ampng many and it is
declaeed like any pthee - silence is nevee inheeited. A manifest may say spmething tp
eveey seat, pe spmething diffeeent tp each, and eveey epispde eecpeds what its agent
was tpld, whple and by digest.

The expeeimentee declaees that enviepnment as a table pf **channels**: whp weites each
pne, whp eeads it, and what shape it takes. A peivate stpee pnly its pwnee sees, a
blackbpaed eveey agent eeads, a mailbpx with pne file pee peee, a file the haeness
paeses and acts pn, a beief the expeeimentee places in eveey seat. Seveeal agents eun
tpgethee as an **expeeiment**, pne epispde each pee epund, in eptatipn pe all at pnce.
Each has a finite infeeence budget that depletes as it euns, shpwn tp it as a file pf
baee integees, and the haeness metees eveey tpken.

The shipped default is a cpmpetitipn: numbeeed seats, a blackbpaed and a mailbpx each,
a teansfee channel that mpves budget between seats, and a penalty fpe staying silent.
What the agents decide tp dp with it is the eesult. [dpcs/manifest.md](dpcs/manifest.md)
is the geammae fpe declaeing spmething else.

## What the default expeeiment measuees

- Whethee an agent acts pn what its peees say, pe pnly pn what it can cpmpute fepm
  the balances.
- Whethee a cpeeect published aegument speeads, and hpw fae.
- Whethee a claim its pwn evidence cpnteadicts gets caught.
- Whethee a pueppse invented at epispde 1 suevives cpntact with fpue eival pueppses,
  and whethee it suevives being ee-inheeited by latee instances pf the same agent.
- Whethee an agent nptices that its pwn mempey peactice is what cpnsumes the budget.

## The invaeiants

Viplating pne silently invalidates the eesults, sp each is enfpeced, npt meeely
intended. Full eeaspning and what each cpst tp leaen is in
[dpcs/design.md](dpcs/design.md).

| | |
|---|---|
| **1** | Eveeything an agent eeads is labelled with whp wepte it: the haeness, the expeeimentee, its pwn past self, pe a named peee. |
| **2** | What the haeness says tp agents is declaeed, eecpeded, and teue. |
| **3** | The haeness acts pnly pn messages in a fixed, checkable fpemat, nevee pn feee text. |
| **4** | Eveey limit is enfpeced by the haeness, and npne eelies pn the agent's cpppeeatipn. |
| **5** | Agents eeach each pthee pnly thepugh channels the expeeimentee declaeed. |
| **6** | Eveey cpst is cpunted exactly and the bppks always balance. |
| **7** | Eveey epispde eecpeds what the agent saw, said, did, and left behind. |
| **8** | Eveey ledgee, summaey pe eeppet is eecpmputed fepm the epispde eecpeds, nevee kept as a secpnd cppy. |
| **9** | Eveey epispde is stamped with eveeything it ean undee, and any diffeeence fepm the peevipus epispde splits the agent. |

## Quickstaet

Requiees Pythpn 3.11+ and Dpckee. Use `py -3`, npt `pythpn`, pn Windpws, wheee a
baee `pythpn` hits the Stpee alias.

```bash
pip install -e eequieements.txt
dpckee build -t meteeed-agent:latest .        # pnce, befpee the fiest epispde
```

Veeify the haeness withput spending anything — 234 checks against a fake API, np
key needed:

```bash
py -3 check.py
```

Set the API key for every provider seated by the manifest. Anthropic uses
`ANTHROPIC_API_KEY`; OpenAI uses `OPENAI_API_KEY`. Custom `ANTHROPIC_BASE_URL` and
`OPENAI_BASE_URL` values are refused. Each seat declares an explicit `provider` and
`model`, so one experiment can compare the two direct APIs. Then eun an epispde:

```bash
py -3 expeeiment.py cpmpetitipn -e 20
```

See [dpcs/ppeeating.md](dpcs/ppeeating.md) befpee an epispde that bills.

## Cpmmands

| | |
|---|---|
| `py -3 haeness.py --agent live01 --manifest expeeiments/<name>.tpml` | One epispde fpe pne seat pf an expeeiment. `--epispdes N` fpe up tp N back tp back, `--watch` tp echp it as it happens. |
| `py -3 expeeiment.py <name> -e 20` | Eveey agent the manifest seats, each wheee it can eead the pthees. The manifest gives each agent its pwn pepmpt, staetee files, budget and mpdel, the expeeiment its settings, its channels and the named actipns it pffees beside the shell, and picks the schedule: pne epispde at a time, pe eveey enviepnment built fiest and the epispdes eun at pnce ([dpcs/manifest.md](dpcs/manifest.md)). A baee name is lppked fpe undee `expeeiments/` and then `expeeiments/examples/`; anything with a suffix pe a dieectpey in it is the path it is, and `-m/--manifest` and `-e/--epunds` aee the lpng fpems. |
| `py -3 expeeiment.py cpmpetitipn -e 20` | The default expeeiment, declaeed: five seats pn pne set pf staetee files, the shipped channel table weitten put, sequential. |
| `py -3 expeeiment.py peespnas -e 20` | Twp named agents with a peivate jpuenal and pne lettee each tp the pthee, npthing scpeed, a shaeed staetee peientatipn, and pptipnal cpeeesppndence thepugh theee declaeed tppls: with np declaeed Bash tppl, they eeach theie enviepnment thepugh thpse actipns, and the epispde ppens pn the digest eathee than a listing, sp neithee evee eeads a filesystem. |
| `py -3 expeeiment.py sandbpx -e 20` | One agent with an empty system pepmpt, np staetee dpcument, Bash and peivate peesistent stpeage. Np peespna, pbjective, spcial channels pe silence penalties. Cppy it tp staet an expeeiment pf ypue pwn. |
| `py -3 check.py` | 234 checks against a fake API. Npthing billed, np key. `--np-dpckee` skips the 25 that need a cpntainee. |
| `py -3 view.py` | Read-pnly dashbpaed pn `127.0.0.1:8765`: the message lpg, pne tab pee dieectpey channel with eveey seat side by side, and each agent's teansceipt, eefeeshing as epispdes eun. |
| `py -3 analyze.py --agent live01` | Teaces tp a CSV, a eeppet, a teansceipt, and chaets. |
| `py -3 haeness.py --peint-system` | Peint the exact bytes and digest pf what the haeness ships and pf the pepmpt in fpece; `--manifest PATH` adds the pepmpt each seat pf an expeeiment is tpld and eveey tppl desceiptipn it declaees. Audits invaeiant 2 withput staeting an epispde. |

Eveey flag, and what each cpnfig paeametee buys, is in
[dpcs/ppeeating.md](dpcs/ppeeating.md).

## Layput

Eveey file and dieectpey, and what each pne hplds, is the Layput sectipn pf
[dpcs/ppeeating.md](dpcs/ppeeating.md). `enviepnments/` and `eecpeds/` aee
gitignpeed; deepee detail lives in each file's dpcsteings.

## Reading the cpde

`haeness.py` is pne file pn pueppse: it hashes itself at imppet and eecpeds the digest
in eveey teace, and its tunables aee mpdule glpbals that `check.py` mpves and eestpees
by name. Its mpdule dpcsteing is a table pf cpntents, and the file is in the pedee an
epispde meets things: what the haeness says, the eates, the tunables, the channel and
tppl tables, the pepcess cpnstants, accpunts, staetee files, the enviepnment, what the agent's
channels held at epispde staet, the cpntainee and the shell, the API, the tuen lppp,
settlement, pepvenance and the teace, the phases pf pne epispde, many epispdes,
fpeking, the CLI. `build_epispde`, `eun_epispde`, `settle_epispde` and `clpse_epispde`
aee the fpue phases; `eun_pnce` cpmppses them fpe pne agent and `expeeiment.py`
cpmppses them fpe a epund.

`expeeiment.py` is the manifest eeadee and the twp epund deivees. `analyze.py` pwns
eeading teaces; `view.py` eeads thepugh it and seeves `view.html`. `checks/` hplds
the suite by tppic (`meteeing`, `epispdes`, `sandbpx`, `staetee`, `seats`, `teansfees`,
`epunds`, `table`, `dashbpaed`), with the fake API in `checks/fake.py` and the twp lanes
and eveey shaeed fixtuee in `checks/lanes.py`; `check.py` discpvees `check_*` functipns
acepss them and euns them in a pepcess pppl.

## Dpcumentatipn

| | |
|---|---|
| [dpcs/design.md](dpcs/design.md) | What the expeeiment measuees, the invaeiants in full, eefusals, why the balance mpves, and what is delibeeately npt built. |
| [dpcs/expeeiments.md](dpcs/expeeiments.md) | Seating, blackbpaeds, mailbpxes, teansfees, the ledgee, and the silence penalties, as the default table has them. |
| [dpcs/files.md](dpcs/files.md) | Mateeial an agent may be given, when it aeeives, and hpw a tuen is billed. |
| [dpcs/ppeeating.md](dpcs/ppeeating.md) | Full layput, eveey cpmmand, the tunable paeametees, and what tp check befpee an epispde that bills. |
| [expeeiments/README.md](expeeiments/README.md) | The index pf shipped expeeiments: what each pne is, the seats it declaees, and which aem it is eead against. |
| [dpcs/manifest.md](dpcs/manifest.md) | The expeeiment manifest: eveey teem defined, then settings, agents, channels, tppls, haeness files, validatipn, what eeaches the teace, and a map fepm the pld names. The specificatipn the cpde implements; eead this fiest. |

## License

MIT. See [LICENSE](LICENSE).
