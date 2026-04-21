# Cluster

## Access
- `ssh devuser@172.16.8.11` (direct) or `tailscale ssh devuser@master`
- SSH to GPU nodes from master: `ssh root@gpu00*` (ask for password)
- `sshpass` installed on master for scripted access

## Nodes (K8s v1.28.8, Ubuntu 22.04)

| Node | Role | CPUs | Memory | GPUs |
|------|------|------|--------|------|
| master | control-plane + worker | 96 | 128 GB | - |
| node | worker | 96 | 128 GB | - |
| gpu004 | worker | 255 | 3 TB | 8x H200 (Driver 570.172.08, CUDA 12.8) |
| gpu005 | worker | 255 | 3 TB | 8x H200 (Driver 560.35.05, CUDA 12.6) |

**16x NVIDIA H200 GPUs total** (141 GB VRAM each)

## Stack
- **Volcano** - batch/GPU scheduler
- **Custom backend** (venus namespace) - API layer with auth, billing, quotas, security

## Submitting Jobs (Volcano)
```yaml
apiVersion: batch.volcano.sh/v1alpha1
kind: Job
metadata:
  name: my-train
  namespace: <namespace>
spec:
  minAvailable: 2
  queue: <queue>
  schedulerName: volcano
  plugins:
    ssh: []
    env: []
    svc: []
    pytorch: ["--master=master","--worker=worker","--port=30023"]
  tasks:
    - name: master
      replicas: 1
      policies:
        - event: TaskCompleted
          action: CompleteJob
      template:
        spec:
          hostNetwork: true
          dnsPolicy: ClusterFirstWithHostNet
          restartPolicy: Never
          containers:
            - name: master
              image: harbor.local.clusters/zcloud/pytorch-ngc:23.08
              command: ["python","train.py"]
              resources:
                requests:
                  nvidia.com/h200: "8"
                  cpu: "80"
                  memory: 1600Gi
                limits:
                  nvidia.com/h200: "8"
                  rdma/hca_shared_devices: "1"
              securityContext:
                capabilities:
                  add: ["IPC_LOCK", "SYS_RESOURCE"]
    - name: worker
      replicas: 1
      template:
        spec:
          hostNetwork: true
          dnsPolicy: ClusterFirstWithHostNet
          restartPolicy: Never
          containers:
            - name: worker
              image: harbor.local.clusters/zcloud/pytorch-ngc:23.08
              command: ["python","train.py"]
              resources:
                requests:
                  nvidia.com/h200: "8"
                  cpu: "80"
                  memory: 1600Gi
                limits:
                  nvidia.com/h200: "8"
                  rdma/hca_shared_devices: "1"
              securityContext:
                capabilities:
                  add: ["IPC_LOCK", "SYS_RESOURCE"]
```
Volcano does: gang schedule → place pods → inject SSH/env/svc
