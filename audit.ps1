# Runs the agent-audit driver against this repository; flags pass through (see the plugin README).
$plugin = $env:AGENT_AUDIT_HOME ?? 'C:\source\repos\agent-audit'
& (Join-Path $plugin 'driver/agent-audit.ps1') -Repo $PSScriptRoot @args
