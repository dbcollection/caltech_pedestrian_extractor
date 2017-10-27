#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""
Extract images (.seq to .jpg) and annotation files (.vbb to .json)
from the Caltech Pedestrian Dataset.
"""


from __future__ import print_function, division
import struct
import os
import json
import argparse
from collections import defaultdict
from scipy.io import loadmat


def read_header(ifile):
    """
    Read the header of a .seq file.
    """
    feed = ifile.read(4)
    norpix = ifile.read(24)
    version = struct.unpack('@i', ifile.read(4))
    length = struct.unpack('@i', ifile.read(4))
    assert(length != 1024)
    descr = ifile.read(512)
    params = [struct.unpack('@i', ifile.read(4))[0] for i in range(9)]
    fps = struct.unpack('@d', ifile.read(8))
    ifile.read(432)
    image_ext = {100: 'raw', 102: 'jpg', 201: 'jpg', 1: 'png', 2: 'png'}
    return {
        'w': params[0],
        'h': params[1],
        'bdepth': params[2],
        'ext': image_ext[params[5]],
        'format': params[5],
        'size': params[4],
        'true_size': params[8],
        'num_frames': params[6],
    }


def read_seq(path):
    """
    Read .seq files to a list.
    """
    assert path[-3:] == 'seq', path
    ifile = open(path, 'rb')
    params = read_header(ifile)
    bytes = open(path, 'rb').read()

    imgs = []
    extra = 8
    s = 1024
    for i in range(params['num_frames']):
        tmp = struct.unpack_from('@I', bytes[s:s + 4])[0]
        img = bytes[s + 4:s + tmp]
        s += tmp + extra
        if i == 0:
            val = struct.unpack_from('@B', bytes[s:s + 1])[0]
            if val != 0:
                s -= 4
            else:
                extra += 8
                s += 8
        imgs.append(img)

    return imgs, params['ext']


def read_vbb(path):
    """
    Read the data of a .vbb file to a dictionary.
    """
    assert path[-3:] == 'vbb'

    vbb = loadmat(path)
    nFrame = int(vbb['A'][0][0][0][0][0])
    objLists = vbb['A'][0][0][1][0]
    maxObj = int(vbb['A'][0][0][2][0][0])
    objInit = vbb['A'][0][0][3][0]
    objLbl = [str(v[0]) for v in vbb['A'][0][0][4][0]]
    objStr = vbb['A'][0][0][5][0]
    objEnd = vbb['A'][0][0][6][0]
    objHide = vbb['A'][0][0][7][0]
    altered = int(vbb['A'][0][0][8][0][0])
    log = vbb['A'][0][0][9][0]
    logLen = int(vbb['A'][0][0][10][0][0])

    data = {}
    data['nFrame'] = nFrame
    data['maxObj'] = maxObj
    data['log'] = log.tolist()
    data['logLen'] = logLen
    data['altered'] = altered
    data['frames'] = defaultdict(list)

    for frame_id, obj in enumerate(objLists):
        if obj.shape[1] > 0:
            for id, pos, occl, lock, posv in zip(obj['id'][0],
                                                 obj['pos'][0],
                                                 obj['occl'][0],
                                                 obj['lock'][0],
                                                 obj['posv'][0]):
                keys = obj.dtype.names
                id = int(id[0][0]) - 1  # MATLAB is 1-origin
                p = pos[0].tolist()
                pos = [p[0] - 1, p[1] - 1, p[2], p[3]]  # MATLAB is 1-origin
                occl = int(occl[0][0])
                lock = int(lock[0][0])
                posv = posv[0].tolist()

                datum = dict(zip(keys, [id, pos, occl, lock, posv]))
                datum['lbl'] = str(objLbl[datum['id']])
                # MATLAB is 1-origin
                datum['str'] = int(objStr[datum['id']]) - 1
                # MATLAB is 1-origin
                datum['end'] = int(objEnd[datum['id']]) - 1
                datum['hide'] = int(objHide[datum['id']])
                datum['init'] = int(objInit[datum['id']])

                data['frames'][frame_id].append(datum)

    return data


def extract_images_video(data_path, save_path):
    """
    Extract + convert .jpg images from .seq file.
    """
    # read images from file
    imgs, ext = read_seq(data_path)

    # save images to file
    for idx, img in enumerate(imgs):
        img_fname = "I{}.{}".format(str(idx).zfill(5), ext)
        img_path = os.path.join(save_path, img_fname)
        with open(img_path, 'wb+') as f:
            f.write(img)


def extract_annotations_video(data_path, save_path):
    """
    Extract + convert annotations to .json from .vbb file.
    """
    # read .vbb file to dict
    data = read_vbb(data_path)

    for i in range(0, data['nFrame']):
        anno_fname = "I{}.json".format(str(i).zfill(5))
        anno_path = os.path.join(save_path, anno_fname)
        try:
            with open(anno_path, 'w') as file_cache:
                json.dump(data['frames'][i],
                          file_cache,
                          sort_keys=True,
                          indent=4,
                          ensure_ascii=False)
        except IOError:
            raise IOError('Unable to open file: {}'.format(anno_path))


def extract_files(data_path, save_path, sets):
    """
    Extract .seq and .vbb files to .jpg and .json.
    """

    sets = sets or ['set00', 'set01', 'set02', 'set03', 'set04',
                    'set05', 'set06', 'set07', 'set08', 'set09',
                    'set10']

    print('')
    print('==> Extract images + annotations from the Caltech Pedestrian Dataset...')

    for j, set_name in enumerate(sets):
        # get set dir
        set_path = os.path.join(data_path, set_name)
        set_path_annot = os.path.join(data_path, 'annotations', set_name)
        set_save_path = os.path.join(save_path, set_name)

        # make save dir
        if not os.path.exists(set_save_path):
            os.makedirs(set_save_path)

        # check if dir exists
        assert os.path.exists(set_path), 'File does not exists: {}'.format(set_path)

        print('\n> Extracting images + annotations from set: {} ({}/{})'
              .format(set_name, j + 1, len(sets)))

        fnames = os.listdir(set_path)
        fnames = [fname for fname in fnames if fname.endswith('.seq')]
        fnames.sort()

        for i, video in enumerate(fnames):

            video_name = os.path.splitext(video)[0]

            print('>> Processing video: {}/{} ({}/{})'
                  .format(set_name, video_name, i + 1, len(fnames)))

            video_path = os.path.join(set_path, video_name + '.seq')
            annot_path = os.path.join(set_path_annot, video_name + '.vbb')
            video_save_path = os.path.join(set_save_path, video_name)
            img_save_path = os.path.join(video_save_path, 'images')
            annot_save_path = os.path.join(video_save_path, 'annotations')

            if not os.path.exists(video_save_path):
                os.makedirs(video_save_path)
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)
            if not os.path.exists(annot_save_path):
                os.makedirs(annot_save_path)

            # ectract images from .seq file to .jpg
            extract_images_video(video_path, img_save_path)

            # extract annotations from .vbb file to .json
            extract_annotations_video(annot_path, annot_save_path)

    print('\n==> Extraction complete.')


def extract_data(data_path, save_path, sets=None):
    """Extract image and annotation data from .vbb and .seq files.

    Parameters
    ----------
    data_path : str
        Directory path of data files.
    save_path : str
        Directory path to store the extracted data.
    sets : str/list/tuple, optional
        List of set names to extract.

    Raises
    ------
    TypeError
        If sets input arg is not a string, list or tuple.
    """
    assert os.path.exists(data_path), "Must provide a valid data path: {}".format(data_path)
    assert save_path, "Must provide a valid storage path: {}".format(save_path)
    if sets:
        if isinstance(sets, str):
            sets = [sets]
        elif isinstance(sets, tuple) or isinstance(sets, list):
            sets = list(sets)
        else:
            raise TypeError('Invalid input type for \'sets\': {}.'.format(type(sets)))

    if not os.path.exists(save_path):
        print('> Saving extracted data to: {}'.format(save_path))
        os.makedirs(save_path)

    # extract images + annotations from .seq and .vbb files
    extract_files(data_path, save_path, sets)


if __name__ == '__main__':
    # parse input args
    parser = argparse.ArgumentParser(description='Caltech Pedestrian Dataset extractor Options.')
    parser.add_argument('-data_path', default='', type=str,
                        help='Dataset directory path.')
    parser.add_argument('-save_path', default='', type=str,
                        help='Store the extracted data into a dir.')
    args = parser.parse_args()

    data_path = args.data_path
    save_path = args.save_path

    assert any(data_path), 'Please insert a valid data path using the -data_path input arg.'

    if not any(save_path):
        save_path = os.path.join(data_path, 'extracted_data')

    extract_data(data_path, save_path)
