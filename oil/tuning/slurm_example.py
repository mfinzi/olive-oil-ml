from oil.tuning.slurmExecutor import SlurmExecutor
import subprocess
import concurrent
import time
import multiprocessing

# "Worker" functions.
def square(n):
    return n * n
def hostinfo(a):
    return subprocess.check_output('uname -a', shell=True).decode()#.split()
def gpustat(a):
    return subprocess.check_output('gpustat', shell=True).decode()#.split()
def cpu_count():
    return multiprocessing.cpu_count()
def example_1():
    """Square some numbers on remote hosts!
    """
    with SlurmExecutor(max_workers=5) as executor:
        futures = [executor.submit(square, n) for n in range(15)]
        for future in concurrent.futures.as_completed(futures):
            print((future.result()))

def example_2():
    """Get host identifying information about the servers running
    our jobs.
    """
    with SlurmExecutor(max_workers=5) as executor:
        futures = [executor.submit(cpu_count) for n in range(5)]
        print('Some cluster nodes:')
        for future in concurrent.futures.as_completed(futures):
            print(future.result())

def example_3():
    """Demonstrates the use of the map() convenience function.
    """
    start = time.time()
    with SlurmExecutor(max_workers=5,clone_session=False) as exc:
        print(''.join(list(exc.map(hostinfo,range(10),chunksize=1))))
    print("Taking a total time of:",time.time()-start)

if __name__ == '__main__':
    #example_1()
    #example_2()
    example_3()
