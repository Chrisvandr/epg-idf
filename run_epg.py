import epg2
import argparse

parser = argparse.ArgumentParser(description='EPG produces EnergyPlus Input Definition (.idf) files.')
parser.add_argument('-type', metavar='-t',choices=['csv', 'script', 'lhd', 'draw'],default='csv',
                   help='Type of run to do: csv, script, lhd or draw (default=csv)')
parser.add_argument('-indir', metavar='-i',default='Debug_runs',
                   help='Directory for csv or script file (default=Debug_runs)')
parser.add_argument('-csv', metavar='-c',default='script.csv',
                   help='csv file name (default=script.csv)')
parser.add_argument('-script', metavar='-s',default='script.txt',
                   help='script file name (default=script.txt)')
parser.add_argument('-build', metavar='-b',default='Semi',
                   help='building type for LHE (default=Semi)')
#parser.add_argument('-nsplit', metavar='-n',default=100, type = int,
#                   help='number of files to split (default=100)')



args = parser.parse_args()

print(args)

if args.type=="draw":
    epg2.draw_building()
elif args.type=="lhd":
    epg2.run_hypercube(indir=args.indir, build_type = args.build)
elif args.type=="script":
    epg2.unpick_script(indir=args.indir,script=args.script)
else:
    epg2.run_csv(indir=args.indir,csvfile=args.csv)
