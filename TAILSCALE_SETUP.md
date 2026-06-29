# Tailscale access plan — DGX Spark server + VSCode-over-SSH from your laptop

Goal: reach the perception server **and** an SSH/VSCode session on the DGX from
anywhere (home wifi, another network, tethered), without port-forwarding, a
public IP, or a VPN appliance. Tailscale puts every device on one private,
encrypted **tailnet** with stable `100.x.y.z` addresses and a name
(`spark-46e5`) that work the same on any network.

Why Tailscale here (vs. raw LAN IP `192.168.45.150`):
- The LAN IP is **DHCP and only routable on that wifi**. The tailnet IP is stable
  and works off-LAN.
- **No router config / port forwarding**; NAT traversal is automatic.
- **WireGuard-encrypted** end-to-end — safe to expose the server + SSH.
- **Tailscale SSH** removes manual SSH key management for the VSCode workflow.

Current DGX state (already verified): `sshd` is **active on :22**, the server
binds `0.0.0.0:8000`, Tailscale is **not yet installed**.

```
  Laptop (VSCode)          tailnet 100.x.y.z (WireGuard)        DGX Spark "spark-46e5"
  ───────────────                                               ──────────────────────
  Remote-SSH  ───────ssh──────────────────────────────────────▶ sshd :22
  browser/client ────http://spark-46e5:8000──────────────────▶  uvicorn :8000 (GB10)
  Windows PC (RealSense client) ─http://spark-46e5:8000──────▶  /perceive
```

---

## Part A — Install Tailscale on the DGX (aarch64 / Ubuntu)

```bash
# 1. Install (official script; supports arm64). Needs sudo.
curl -fsSL https://tailscale.com/install.sh | sh

# 2. Bring it up. --ssh enables Tailscale SSH (see Part D).
sudo tailscale up --ssh --hostname spark-46e5
#    -> prints a https://login.tailscale.com/... URL. Open it, sign in
#       (Google works — same account on every device), approve the machine.

# 3. Confirm + note the tailnet IP (100.x.y.z) and name.
tailscale ip -4           # e.g. 100.92.14.7
tailscale status
```

`tailscaled` installs as a **systemd service and auto-starts on boot** — the DGX
rejoins the tailnet automatically after a reboot. No further action needed.

MagicDNS (on by default for new tailnets) lets you use the name `spark-46e5`
instead of the IP. Enable it once in the admin console if off:
https://login.tailscale.com/admin/dns → "Enable MagicDNS".

---

## Part B — Install Tailscale on your laptop and the Windows PC

- **Laptop** (the one running VSCode) and the **Windows RealSense PC**: install
  the Tailscale app from https://tailscale.com/download and **sign in with the
  same account**. They appear in `tailscale status` on the DGX.
- The Windows PC needs Tailscale too so the perception client can reach the DGX
  off-LAN by `spark-46e5`. On the same wifi it can keep using `192.168.45.150`,
  but the tailnet name works everywhere — prefer it.

---

## Part C — Reach the perception server over the tailnet

The server already binds `0.0.0.0`, so once Tailscale is up it's reachable at the
tailnet address with no change:

```bash
# from the laptop or the Windows PC (any network):
curl http://spark-46e5:8000/health          # MagicDNS name
curl http://100.92.14.7:8000/health         # or the raw tailnet IP
```

Update the Windows client to target the tailnet name:

```python
client = PerceptionClient("http://spark-46e5:8000")     # works on any network
```

Tip: `spark-46e5` resolves via MagicDNS only while Tailscale is running on the
client; fall back to the `100.x` IP if name resolution misbehaves.

---

## Part D — SSH + VSCode Remote-SSH from the laptop

Two options. **Tailscale SSH (recommended)** needs no keys; plain SSH also works.

### Option 1 — Tailscale SSH (no key management)
You already ran `tailscale up --ssh` on the DGX (Part A). Authorize SSH in the
admin console ACLs once (default policy often allows it for your own devices):
https://login.tailscale.com/admin/acls — ensure a rule like:

```jsonc
"ssh": [{
  "action": "accept",
  "src":    ["autogroup:member"],
  "dst":    ["autogroup:self"],
  "users":  ["cosmos", "autogroup:nonroot"]
}]
```

From the laptop (with Tailscale running):
```bash
ssh cosmos@spark-46e5         # auth handled by Tailscale identity, no password/key
```

### Option 2 — Classic SSH key
```bash
# laptop: create a key if needed, then copy it to the DGX
ssh-keygen -t ed25519
ssh-copy-id cosmos@spark-46e5         # or @100.92.14.7
ssh cosmos@spark-46e5                 # confirm it logs in
```

### VSCode Remote-SSH
1. Install the **Remote - SSH** extension (Microsoft) in VSCode on the laptop.
2. Add the host to `~/.ssh/config` on the laptop:
   ```
   Host spark
       HostName spark-46e5          # MagicDNS name (or 100.92.14.7)
       User cosmos
       # ForwardAgent yes
   ```
3. VSCode → Command Palette → **Remote-SSH: Connect to Host… → spark**.
   VSCode installs its server into `~/.vscode-server` on the DGX on first
   connect (one-time, a minute or two).
4. **Open Folder** → `/home/cosmos/claude/01_active_perception_server`. You now
   edit/run on the DGX with the GB10 GPU; the integrated terminal is a DGX shell.

### Port-forward the server through VSCode (optional)
With the Remote-SSH session open, VSCode's **Ports** panel → Forward port `8000`.
Then `http://localhost:8000/health` on the laptop tunnels to the DGX — handy for
poking the API or `/docs` (FastAPI Swagger UI at `http://localhost:8000/docs`)
without exposing anything.

---

## Part E — Run the server as a service (survives logout/reboot)

A unit file is provided: `scripts/perception-server.service`.

```bash
sudo cp scripts/perception-server.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now perception-server
systemctl status perception-server
journalctl -u perception-server -f          # live logs
```

It starts after `tailscaled`, runs uvicorn from the venv as `cosmos`, and
restarts on failure. Edit `Environment=AP_HOST=...` in the unit to change the
bind (see Part F), and add `HF_TOKEN=` there once SAM 3 access is granted.

---

## Part F — Security hardening (recommended)

- **Bind the server to the tailnet only.** Right now `0.0.0.0:8000` is reachable
  by anyone on the wifi LAN. To accept connections *only* over Tailscale, set the
  bind to the DGX's tailnet IP:
  ```bash
  # in the systemd unit:  Environment=AP_HOST=100.92.14.7
  ```
  (Trade-off: same-LAN clients must then also use the tailnet address.) Keep
  `0.0.0.0` if you want plain-LAN access too.
- **Don't use Tailscale Funnel/Serve** for this. Funnel exposes a service to the
  *public* internet; you only need private tailnet access. Avoid it here.
- **Tailscale ACLs**: lock down who can reach `:8000` and SSH. Example tightening
  the server to just your devices:
  ```jsonc
  "acls": [
    { "action": "accept", "src": ["autogroup:member"], "dst": ["spark-46e5:8000,22"] }
  ]
  ```
- **Key expiry**: leave node-key expiry on (default 180 days); you'll re-auth
  occasionally. For an always-on server you may set the DGX key to **not expire**
  in the admin console if that's acceptable for your environment.
- **ufw**: if you later enable a host firewall, allow the tailscale interface:
  `sudo ufw allow in on tailscale0` and `sudo ufw allow in on tailscale0 to any port 8000`.

---

## Part G — Troubleshooting

| symptom | check |
|---|---|
| `spark-46e5` won't resolve | MagicDNS enabled? Tailscale running on the client? Use `100.x` IP. |
| server unreachable over tailnet | `tailscale status` shows DGX online? `AP_HOST` not bound to LAN-only IP? |
| SSH refused | `systemctl status ssh` on DGX; ACL `ssh` rule present for Option 1. |
| VSCode server install hangs | DGX disk/space ok (`df -h ~`); retry "Kill VS Code Server on Host". |
| slow / relayed connection | `tailscale ping spark-46e5` — "direct" good, "via DERP" = relayed but works. |
| works on-LAN, not off-LAN | the client device must also be signed into the tailnet. |

---

## One-time checklist

- [ ] DGX: `curl -fsSL https://tailscale.com/install.sh | sh`
- [ ] DGX: `sudo tailscale up --ssh --hostname spark-46e5`
- [ ] Admin console: MagicDNS on; SSH ACL allows your devices
- [ ] Laptop + Windows PC: install Tailscale, sign in (same account)
- [ ] Laptop: `~/.ssh/config` host `spark`; VSCode Remote-SSH connect
- [ ] DGX: install `perception-server.service`, `enable --now`
- [ ] Verify: `curl http://spark-46e5:8000/health` from the laptop
- [ ] (Optional) bind server to tailnet IP; tighten ACLs
```
