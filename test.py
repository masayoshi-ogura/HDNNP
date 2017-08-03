from mpi4py import MPI

comm = MPI.COMM_WORLD
world_rank = MPI.COMM_WORLD.Get_rank()
world_size = MPI.COMM_WORLD.Get_size()
world_group = MPI.COMM_WORLD.Get_group()
new_group = world_group.Incl(range(8))

newcomm = MPI.COMM_WORLD.Create(new_group)

print world_rank, newcomm.Get_rank(), newcomm.Get_size()
