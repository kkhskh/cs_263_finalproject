#!/bin/bash
# Setup gVisor (runsc) for Docker
# Run as root: sudo ./setup_gvisor.sh

set -e

echo "=== Installing gVisor (runsc) ==="

# Add gVisor repository
curl -fsSL https://gvisor.dev/archive.key | gpg --dearmor -o /usr/share/keyrings/gvisor-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/gvisor-archive-keyring.gpg] https://storage.googleapis.com/gvisor/releases release main" | tee /etc/apt/sources.list.d/gvisor.list > /dev/null

# Install runsc
apt-get update && apt-get install -y runsc

# Configure Docker to use runsc
cat > /etc/docker/daemon.json << 'EOF'
{
    "runtimes": {
        "runsc": {
            "path": "/usr/bin/runsc"
        }
    }
}
EOF

# Restart Docker
systemctl restart docker

# Verify installation
echo ""
echo "=== Verifying gVisor installation ==="
docker run --rm --runtime=runsc hello-world

echo ""
echo "=== gVisor installed successfully ==="
echo "Use with: docker run --runtime=runsc ..."
echo "Or in docker-compose: runtime: runsc"
