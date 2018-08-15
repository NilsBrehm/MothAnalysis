for m in b.multi_tags
    if "SingleStimulus" in m.name:
        np.mean(m.retrieve_data(1,0))


m = b.multi_tags["TagName"]
m.positions[:]
m.positions.unit
m.extents[:]
m.metadata
m.metadata.props
m.metadata.sections
m.references

m.retrieve_data(1,0)[:]

np.arange(0,10.1,3) # (start, steps, end)
np.linspace(0,10,3) # (start, end, #indices)

m.metadata.sections[0].created_at
meta = m.metadata.sections[0]
duration = meta["Duration"]

m.retrieve_feature_data
m.features[4].data # entspricht Amplitude


pos = np.delete(pos,0)




Pass it as a tuple:

print("Total score for %s is %s  " % (name, score))

Or use the new-style string formatting:

print("Total score for {} is {}".format(name, score))

Or pass the values as parameters and print will do it:

print("Total score for", name, "is", score)

If you don't want spaces to be inserted automatically by print, change the sep parameter:

print("Total score for ", name, " is ", score, sep='')

for f in freq_uni:
    for a in amps_uni:
        ids = np.where((frequency == f) & (amplitude == a))[0][:]
        if not ids.any():
            continue
        for j in range(len(ids)):
            spikes[j] = len(mtag.retrieve_data(int(ids[j]), 1)[:])