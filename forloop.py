import numpy as np
from multiprocessing import Pool, shared_memory, cpu_count
import tracemalloc
from time import sleep

default_nprocs = cpu_count()

def create_memory_block(lin_size=512,seed=123,shared=True):

  np.random.seed(seed)
  a = np.random.standard_normal((lin_size,lin_size,lin_size))
  if (shared):
    shm = shared_memory.SharedMemory(create=True,size=a.nbytes)
    sha = np.ndarray(a.shape,dtype=a.dtype,buffer=shm.buf)
    # Copy array into shared memory
    sha[:] = a[:] 
    return sha, shm

  else:
    return a


def slow_mean_shared(shm_name,lin_size,sl):

  shm = shared_memory.SharedMemory(shm_name)
  a = np.ndarray(shape=(lin_size,lin_size,lin_size),dtype='float64',buffer=shm.buf)
  sleep(0.01)
  return a[sl].mean()

def slow_mean(arr,lin_size,sl):
  sleep(0.01)
  return arr[sl].mean()

def distribute(nitems, nprocs=None):
  if nprocs is None:
    nprocs = default_nprocs
  nitems_per_proc = (nitems+nprocs-1)//nprocs
  return [slice(i,min(nitems,i+nitems_per_proc)) for i in range(0,nitems,nitems_per_proc)]


def doit(lin_size=256,shared=True,nprocs=None):

  if nprocs is None:
    nprocs = default_nprocs

  res = np.zeros(nprocs)
  pool = Pool(nprocs)
  if shared:
    data, shm = create_memory_block(lin_size,shared=True)
  else:
    data = create_memory_block(lin_size,shared=False)
  slices = distribute(lin_size,nprocs)

  tracemalloc.start()
  if (shared):
    results = [pool.apply_async(slow_mean_shared,(shm.name,lin_size,sl)) for sl in slices]
  else:
    results = [pool.apply_async(slow_mean,(data,lin_size,sl)) for sl in slices]
  # Copy results into single array
  for i,r in enumerate(results):
    res[i] = r.get()

  print("Allocated memory: current %d peak %d"%tracemalloc.get_traced_memory())
  print (res)
  tracemalloc.stop()



