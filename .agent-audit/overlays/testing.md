# Test attribution

The suite is check.py, which discovers `check_*` functions across checks/<topic>.py and runs them in a process pool; pytest collects none of them. checks/fake.py is the fake API, and checks/lanes.py holds the fixtures and the local and container lanes. Container checks skip when Docker is down. Tunables are harness.py module globals that checks set and restore by name. Line coverage is not collected.
