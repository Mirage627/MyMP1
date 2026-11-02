import h5py
f = h5py.File('data/robomimic/datasets/stack_d1/stack_d1_abs.hdf5', 'r')
print(list(f['data/demo_0/obs'].keys()))
f.close()


f = h5py.File('data/robomimic/datasets/core/stack_d1.hdf5', 'r')
print(list(f['data/demo_0/obs'].keys()))
f.close()