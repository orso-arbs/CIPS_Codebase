import pandas as pd
import numpy as np

M_state = 56255808136 / 1073741824
print("Storage per state (GB):", M_state)

N_States = 134
print("N_States:", N_States)

M_all_states = N_States * M_state
print("M_all_states (GB):", M_all_states)

M_scratch_max = 2.5 * 1024
print("M_scratch_max (GB):", M_scratch_max)

M_scratch_80percent = 0.8 * M_scratch_max
print("M_scratch_80percent (GB):", M_scratch_80percent)     

N_scatch_max = M_scratch_max / M_state
print("N_scatch_max:", N_scatch_max)

##### scratch properties

columns = ["location",
           "states",
           "#states", 
           "total memory available [GB]",
           "total memory used by states [GB]",
           "percent free when used",
           ]

# Create an empty DataFrame with the specified columns
scratch_properties_df = pd.DataFrame(columns=columns)

n_i = 45

n_orsob = n_i
M_orsob = n_orsob * M_state
l_orsob = M_orsob / M_scratch_max
new_row = {
    "location": "/cluster/scratch/orsob/orsoMT_orsob",
    "total memory available [GB]": 2.5 * 1024,
    "total memory used by states [GB]": M_orsob,
    "percent free when used": (1 - l_orsob) * 100,
    "#states": n_orsob,
    "states": f"0 - {n_orsob - 1}",
}
new_row_df = pd.DataFrame([new_row])  # Convert the new_row dict to a DataFrame
scratch_properties_df = pd.concat([scratch_properties_df, new_row_df], ignore_index=True)

n_doancea = n_i
M_doancea = n_doancea * M_state
l_doancea = M_doancea / M_scratch_max
new_row = {
    "location": "/cluster/scratch/doancea/orsoMT_doancea",
    "total memory available [GB]": 2.5 * 1024,
    "total memory used by states [GB]": M_doancea,
    "percent free when used": (1 - l_doancea) * 100,
    "#states": n_doancea,
    "states": f"{n_orsob} - {n_orsob + n_doancea - 1}",
}
new_row_df = pd.DataFrame([new_row])  # Convert the new_row dict to a DataFrame
scratch_properties_df = pd.concat([scratch_properties_df, new_row_df], ignore_index=True)


n_aolareanu = n_i - 1
M_aolareanu = n_aolareanu * M_state
l_aolareanu = M_aolareanu / M_scratch_max
new_row = {
    "location": "/cluster/scratch/aolareanu/orsoMT_aolareanu",
    "total memory available [GB]": 2.5 * 1024,
    "total memory used by states [GB]": M_aolareanu,
    "percent free when used": (1 - l_aolareanu) * 100,
    "#states": n_aolareanu,
    "states": f"{n_orsob + n_doancea} - {n_orsob + n_doancea + n_aolareanu - 1}",
}
new_row_df = pd.DataFrame([new_row])  # Convert the new_row dict to a DataFrame
scratch_properties_df = pd.concat([scratch_properties_df, new_row_df], ignore_index=True)

print(scratch_properties_df.T)
print("\n\n")


print("\nScratch: D")
n_D1 = N_States - n_orsob - n_doancea - n_aolareanu
n_D1 = 0
n_D2max = M_scratch_max/M_state - n_D1
print("n_D1:", n_D1, "\tn_D2max:", n_D2max)

n_D2 = 45
n_D = n_D1 + n_D2
M_D = n_D * M_state
l_D = M_D / M_scratch_max
print("n_D1:", n_D1, "\tn_D2:", n_D2, "\tn_D:", n_D)
print(
    "states:", n_D,
    "\nmemory used [GB]:", M_D,
    "\npercent free:", 1-l_D
    )


##### selection of states

print("\n Full Set: Distribution in scratches orsob, doancea, aolareanu")

print("Scratch: orsob"
    "\nstates:", 1, " to ",  n_orsob)
print("Scratch: doancea"
      "\nstates:", n_orsob + 1, " to ",  n_orsob + n_doancea)

print("Scratch: aolareanu"
      "\nstates:", n_orsob + n_doancea + 1, " to ",  n_orsob + n_doancea + n_aolareanu)

# print("Scratch: D"
#       "\nstates in:", n_orsob + n_doancea + n_aolareanu + 1, " to ",  n_orsob + n_doancea + n_aolareanu + n_D1)





a = 134.0
s = 3.0
c = 1.0
n = 1
print("\n Selection Set: Distribution in scratch D \n selection starting at", c, "incrementing by", s)
while c <= a:
    print(c, "\t", n)
    c = c + s
    n = n + 1