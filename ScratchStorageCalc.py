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

n_i = 45

print()
n_Orso = n_i
M_Orso = n_Orso * M_state
l_Orso = M_Orso / M_scratch_max
print(
    "\nScratch: Orso"
    "\n# states:", n_Orso,
    "\nmemory used[GB]:", M_Orso,
    "\npercent free:", 1-l_Orso
    )

n_B = n_i
M_B = n_B * M_state
l_B = M_B / M_scratch_max
print(
    "\nScratch: B"
    "\n# states:", n_B,
    "\nmemory used [GB]:", M_B,
    "\npercent free:", 1-l_B
    )


n_C = n_i - 1
M_C = n_C * M_state
l_C = M_C / M_scratch_max
print(
    "\nScratch: C"
    "\n# states:", n_C,
    "\nmemory used [GB]:", M_C,
    "\npercent free:", 1-l_C
    )

print("\nScratch: D")
n_D1 = N_States - n_Orso - n_B - n_C
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

print("\n Full Set: Distribution in scratches Orso, B, C")

print("Scratch: Orso"
      "\nstates:", "1", " to ",  n_Orso)

print("Scratch: B"
      "\nstates:", n_Orso + 1, " to ",  n_Orso + n_B)

print("Scratch: C"
      "\nstates:", n_Orso + n_B + 1, " to ",  n_Orso + n_B + n_C)

# print("Scratch: D"
#       "\nstates in:", n_Orso + n_B + n_C + 1, " to ",  n_Orso + n_B + n_C + n_D1)





a = 134.0
s = 3.0
c = 1.0
n = 1
print("\n Selection Set: Distribution in scratch D \n selection starting at", c, "incrementing by", s)
while c <= a:
    print(c, "\t", n)
    c = c + s
    n = n + 1