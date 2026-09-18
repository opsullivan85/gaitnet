# Third-party code in gaitnet-mpc

| Component | Location | License | Upstream |
|---|---|---|---|
| rl-mpc-locomotion (Python MPC, `mpc_osqp.cc`) | `src/gaitnet_mpc/mpc`, `cpp/` | MIT (`LICENSE`) | https://github.com/silvery107/rl-mpc-locomotion, via https://github.com/opsullivan85/rl-mpc-locomotion |
| OSQP 0.6.0 | `extern/osqp` | Apache-2.0 | https://github.com/osqp/osqp |
| qpOASES 3.2 | `extern/qpoases` | LGPL-2.1 (`extern/qpoases/LICENSE.txt`) | https://github.com/coin-or/qpOASES |
| Eigen 3.3.9 | `extern/eigen3` (git submodule) | MPL-2.0 | https://gitlab.com/libeigen/eigen |
