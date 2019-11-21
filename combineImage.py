# Check whether image has both upper and lower or whole
    upper = 0
    lower = 0
    whole = 0
    for x in r['class_ids']:
        t = x-1
        if t<5:
            upper += 1
        elif t<9:
            lower += 1
        elif t<13:
            whole += 1
    if whole>0 or (upper>0 and lower>0):
        data.append(r)
        url_data.append(url)
        url_index.append(i)
    
    print(url_index, " : url_index")
    #for t in range(rois[t])
    #scipy.misc.imsave('outfile.jpg', rois)
    print(img.shape, " : image")

    #combine image
    
    i = 0
    whole = 0
    
    for x in r['class_ids']:
        t = x-1
        if t<5:
            u = img[rois[i][0]:rois[i][2], rois[i][1]:rois[i][3]]
        elif t<9:
            l = img[rois[i][0]:rois[i][2], rois[i][1]:rois[i][3]]
        elif t<13:
            whole += 1
            w = img[rois[i][0]:rois[i][2], rois[i][1]:rois[i][3]]
        i = i + 1
        
    if whole>0:
        j = resize_url_image(w)
        k = Image.fromarray(j)
        k.save("output.jpeg")
    else:
        up = resize_url_image(u)
        low = resize_url_image(l)
        combine = np.append(up,low, axis = 0)
        k = Image.fromarray(combine)
        k.save("output.jpeg")
