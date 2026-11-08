exec(open('an1.py').read().split('samp = np')[0])
tk = [x for x in L['take'] if reg(x[0]) == 'vol' and x[5] > 0]
q = np.array([x[4] for x in tk]); room = np.array([x[5] for x in tk])
part = q < room
print('vol take decisions with room>0: %d; avail<room in %.2f; unused room sum %d lots (sent %d)' % (len(tk), part.mean(), (room - q)[part].sum(), np.minimum(q, room).sum()))
print('room quantiles when partial', np.percentile(room[part], [25, 50, 75]), 'avail', np.percentile(q[part], [25, 50, 75]))
