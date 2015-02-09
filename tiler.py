"""Run like python tiler.py myimage.jpg.
Tiler produces myimage.txt and myimage.txt.png."""

import sys, getopt
import matplotlib as mpl
#http://stackoverflow.com/questions/25561009/how-do-you-i-use-mandarin-charecters-in-matplotlib
#mpl.use("pgf")  #I'm not there with Chinese yet.
import matplotlib.pyplot as plt, cStringIO, numpy as np, IPython.display as dis, base64
from PIL import Image
from io import BytesIO

#%matplotlib inline

#Get the pixelated versions of characters.
def charDat(aChar, wgt, fs, **kwargs):
    """Pass an character, weight='normal' or 'bold', and fontsize.  Get back a numpy array.
    Can also pass fontname."""
    plt.clf();#clean up
    plt.cla();
    fontname =  kwargs.get('fontname', 'Bitstream Vera Sans')
    fig = plt.gcf(); #The current figure
    mydpi=80.;
    fs = float(fs); #fontsize
    fig.set_size_inches(fs/mydpi, fs/mydpi); 
    plt.axis('off'); #Hide the axes
    #The char wasn't filling the box so I scaled it by eyeballing it.
    plt.text(0,0.0, aChar, fontsize=fs*1.2, weight=wgt);# weight='bold' if you want
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #Save plot as image encoded as text.
    sio = cStringIO.StringIO();
    plt.savefig(sio, dpi= mydpi, format="png");
    plt.close();
    s = sio.getvalue().encode('base64').strip();

    #Get the image to be an array of its values (RGB+a 4th for transparency I think.)
    im = Image.open(BytesIO(base64.b64decode(s)));
    dat = list(im.getdata())
    datarr = np.array(dat).reshape(list(im.size)+[4]); #+4 for the 4 elements of the png
    return datarr[:,:,0] #Will be BW, so just take the red channel (0)

def getBase64FromArray(arr):
    """Show the image from a numpy array.  Use Python's display features. 
This is for checking how arrays look."""
    arr = arr.astype('B');
    if len(arr.shape)==3:
        im = Image.fromarray(arr, 'RGB')
    else:
        im = Image.fromarray(arr)
    buff= cStringIO.StringIO()
    im.save(buff, format="PNG")
    b64 = base64.b64encode(buff.getvalue())
    buff.close()
    return b64;
    
def saveArrToPNG(arr, fn):
    f = open(fn, 'w');
    im = Image.fromarray(arr.astype('B'))
    im.save(f, format='PNG')
    f.close();
    
class tileSet(object):
    
    def __init__(self, **kwargs):
        """Pass:
characters = (list of characters or string), will be turned into images
width:    default 12, width of tiles, fontsize will be something like it


Attributes:
tile: a list of the tiles, a numpy array of numpy arrays.  Allow for them to be in color.

To do:
pass images=[fns]: a list of fileames of images, could do but not there yet
Make an iterator. Where t is a tileSet, want t[3] to give me t.tile[3]. Would be nice.
Check that all tiles are the same size when import the filenames.  The characters are forced to be the same size.
"""
        self.characterTiles = True; #Will be false when / if use photos for the tiles.
        self.characters = kwargs.get('characters', [str(chr(c)) for c in range(32,127)]); #default chars.
        self.dim = (kwargs.get('width',12), kwargs.get('width',12)); #Works for characters.  Would have to check images.
        self.tile = np.array([charDat(c, 'normal', self.dim[0]) for c in self.characters]).astype('B')

class tiler(object):
    def __init__(self, fn, tiles, **kwargs):

        """Pass 
1. source filename
2. tiles to use in making the picture: a tileSet object
3. optional named args:
    transpose=True or False (default) to transpose image
    power = 0 to white out to 1 to leave alone to >1 to make darker (up to +infinity to black at limit)

    Returns list:
    * Input image displayed with white-out, transpose applied
    * Resulting image of letters
    * Values underlying the new image
    * characters in the list that matched, by block (which characters are used in the list: index)
    * the text of the image

    """
        self.fn = fn;
        self.tiles = tiles;
        
        #Read in the data from the source image
        im = Image.open(fn);
        data = np.array(list(im.getdata()));
        data = data.reshape([im.size[1], im.size[0], len(data[0])]);
        im.close();
        if kwargs.get('transpose',False):
            data = np.transpose(data, (1,0,2)); #Some test images needed transposing, dunno why
        self.dataOriginal = data; 
        #Force black and white target.  Fix later when have color tiles.
        self.dataTarget = 255 * ((np.mean(data, -1) * 1. / 255.) ** kwargs.get('power', 1.0) ) #Exponentiate fraction of brighness. [0-1) brigher 1 same, (1,inf+) darger

    def matchTiles(self):
        """Match the tiles to the portions of the image.  Set:
        matches: an array of the indices of tiles taht match
        dataText: the numerical data of the image, which can then be written out."""
        dim = self.tiles.dim;
        
        #Check each block.
        targdat = self.dataTarget
        newdat = targdat.copy(); #Will overwrite all this
        #The numbers of the tiles that matched to each place:
        matches = np.zeros((1+targdat.shape[0]/dim[0], 1+targdat.shape[1]/dim[1])); #The tiles I match
        
        for i0 in range(0, targdat.shape[0], dim[0]):
            for i1 in range(0, targdat.shape[1], dim[1]):
                ablk = newdat[i0:i0+dim[0], i1:i1+dim[1]]; # A block in the target data.
                #Get the dot products of this block with each character.  Pick the largest.
                n = ablk.shape; #For the last row, column that will get truncated.  Need to know size.
                #dist is the distance from the letter to the image, for each letter in the palette.
                
                #Not a great use of the numpy lin alg features here, but works.
                
                dist = list(np.array([np.linalg.norm((ablk - cod[:n[0], :n[1]]).flatten()) 
                                      for cod in self.tiles.tile]));
                j = dist.index(min(dist)) #The position of the (first) best match.  
                                        # Fix: @ random in prop to match quality so don't get big patches of same letter.
                matches[i0/dim[0],i1/dim[1]] = j; #Save the one that matched.

                #Put that letter in, only up to the lower or rightmost edge.
                newdat[i0:i0+n[0], i1:i1+n[1]] = self.tiles.tile[j, :n[0], :n[1]];
        self.matches = matches; 
        self.dataText = newdat;

    def show(self):
        return dis.HTML('<table><tr>%s</tr></table>'%'\n'.join(['<td><img src="data:image/png;base64,%s"></img></td>'%getBase64FromArray(d)
                        for d in [self.dataOriginal, self.dataTarget, self.dataText]]))

    def save(self):
  
        outfn = self.fn; #assume no dot.  Wrong probably but doesn't hurt.
        if '.' in outfn:
            outfn = '.'.join(outfn.split('.')[:-1]) #Strip extension 
        outfn += '.txt'
        #Save the characters, if there are characters.
        if self.tiles.characterTiles:
            txt = '\n'.join([''.join([self.tiles.characters[int(c)] for c in r]) for r in self.matches])
            f = open(outfn, 'w');
            f.write(txt);
            f.close();
        #Save the image.
        saveArrToPNG(self.dataText, outfn+'.png');

# For below see http://www.artima.com/weblogs/viewpost.jsp?thread=4829
def main():
    # parse command line options
    try:
        opts, args = getopt.getopt(sys.argv[1:], "h", ["help"])
    except getopt.error, msg:
        print msg
        print "for help use --help"
        sys.exit(2)
    # process options
    for o, a in opts:
        if o in ("-h", "--help"):
            print __doc__
            sys.exit(0)

    # process arguments
    fn = args[0];
	
    ts = tileSet() #just use defaults
    t = tiler(fn, ts)
    t.matchTiles();
    t.save();

if __name__ == "__main__":
    main()
