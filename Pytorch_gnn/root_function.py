import ROOT
import os,string,copy
def findstatcoords(statbox):
    """Finds the coordinates of the given statbox."""
    x2 = statbox.GetX2NDC()
    y2 = statbox.GetY2NDC()
    x1 = statbox.GetX1NDC()
    y1 = statbox.GetY1NDC()
    return (x1,y1,x2,y2)

def getstatbox(histo):
    """Gets the statbox object from the histo."""
    list_func = histo.GetListOfFunctions()
    statbox = list_func.FindObject('stats')
    try:
        statbox.GetName();
        return statbox
    except ReferenceError:
        print ('getstatbox failed, please check that the histogram is drawn before calling this function.')
        raise


def leftordown(boxpos,width,height):
    """Given a maximum width or height, returns whether the next statbox should be left of the current one, or on the next line."""
    bp1 = (boxpos[0]+1,boxpos[1]+1)
    if bp1[0] <= width:
        return 'left'
    elif bp1[1] <= height:
        return 'down'
    else:
        raise IndexError('bp1 exceeds gooddimensions!')

def movestatbox(statbox,x1,y1,x2,y2):
    """Moves the statbox to a new location.  
    Uses the NDC coordinates where the canvas is 1 wide and tall, (0,0) is the bottom left."""
    statbox.SetX1NDC(x1)
    statbox.SetX2NDC(x2)
    statbox.SetY1NDC(y1)
    statbox.SetY2NDC(y2)


def colourize_statbox(histo):
    """Takes a histogram (already drawn) and makes the statbox the same colour as the line or marker."""
    statbox = getstatbox(histo)
    
    marker_colour = histo.GetMarkerColor()
    line_colour = histo.GetLineColor()
    if not marker_colour == line_colour:
        raise ValueError('Marker and line colour not the same, check colourizing code.')
    
    statbox.SetTextColor(marker_colour)
    return 1
    
def gooddimensions(n):
    """Returns good histogram layout for a split canvas."""
    width = findleqsquare(n)
    for height in range(0,n+1):
        if height*width >= n:
            return width,height
    raise ValueError('Could not find good dimensions.')

def findleqsquare(n):
    """Finds the largest integer whose square is less than or equal to n."""
    for i in range(1,n):
        if i*i > n:
            return i-1
    return 1
    
def biggest(histos):
    """When multiple histos are draw, ideally the tallest one is drawn first, with he others using the same axes.  This function returns the key of the tallest histogram.  The input histos should be a dict of histograms."""
    return max(histos,key=lambda hist: histos[hist].GetMaximum())
    
def draw_mult_stack(histos_dict,title=None,img_name='',path=''):
    """Draws multiple histos on the same pad, stacked on top of each other."""

    #OptStat is set by "ksiourmen"
    #kurtosis, skewness, integral, n_overflows, n_underflows,
    #rms, mean, n_entries, name
    #k,s,r,m can be 0, 1 or 2 (to get error)
    #all others can be 0 or 1
    #do not include leading 0s, else get hexed.
    oldstyle = ROOT.gStyle.GetOptStat()
    #ROOT.gStyle.SetOptStat(1201) #RMS, Mean with error, and name
    # ROOT.gStyle.SetOptStat(110011) # both overflows, n_entries, name.
    oldH = ROOT.gStyle.GetStatH()
    ROOT.gStyle.SetStatH(0.10)

    histos = copy.deepcopy(histos_dict)
    width,height = gooddimensions(len(histos))

    c1 = ROOT.TCanvas()
    c1.SetLogy()
    if title:
        c1.SetTitle(title)
    count = 1

    bigkey = biggest(histos)
    histos_keys = histos.keys()
    histos_keys = sorted(list(histos_keys))
    # histos_keys.remove(bigkey)
    # histos_keys.insert(0,bigkey)

    boxpos = ()
    first = True
    for h in histos_keys:
        if first:
            histos[h].Draw()
            histos[h].SetTitle(title)
            ROOT.gPad.Update()
            statbox = getstatbox(histos[h])
            Ox1,Oy1,Ox2,Oy2 = findstatcoords(statbox)
            boxpos = (1,1)
            first = False
        else:
            histos[h].SetTitle(title)
            histos[h].Draw('sames')
            ROOT.gPad.Update()
            go = leftordown(boxpos,width,height)
            statbox = getstatbox(histos[h])
            if go == 'left':
                movestatbox(statbox,x1-(x2-x1),y1,x2-(x2-x1),y2)
                boxpos = (boxpos[0]+1,boxpos[1])
            elif go == 'down':
                movestatbox(statbox,Ox1,y1-(y2-y1),Ox2,y2-(y2-y1))
                boxpos = (1,boxpos[1]+1)
            ROOT.gPad.Modified()

        x1,y1,x2,y2 = findstatcoords(statbox)
        colourize_statbox(histos[h])
        ROOT.gPad.Modified()
        ROOT.gPad.Update()
        count += 1
    ROOT.gPad.Update()
    legend = ROOT.TLegend(0.1, 0.9, 0.4, 0.8)
    histos_keys = sorted(list(histos_dict.keys()))
    if(len(histos_keys)==2):
        for histkey in histos_keys:
            # histos_dict[histkey].Draw('same')
            if 'trad' in histkey:
                legend.AddEntry(histos_dict[histkey], 'Analytical pipeline ', 'l')  
            else:
                legend.AddEntry(histos_dict[histkey], 'AI pipeline ', 'l')
    else:
        for histkey in histos_keys:
            if 'trad' in histkey:
                legend.AddEntry(histos_dict[histkey], 'Analytical pipeline ', 'l')  
            elif 'prim'in histkey:
                legend.AddEntry(histos_dict[histkey], 'MC truth ', 'l')  
            else:
                legend.AddEntry(histos_dict[histkey], 'AI pipeline ', 'l')

    legend.Draw()
    ROOT.gPad.Modified()
    # ROOT.gPad.Update()
    count += 1   
    
    c1.SaveAs(f'{path}/{img_name}.png')
 




        