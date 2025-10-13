#! python3

import sys
from svgpathtools import Path, Line, Arc, svg2paths2

from tools.utils import get_svg_size

# sample_rate_mm = 5
sample_rate_mm = 1
path_filename = []
translation = (0, 0)
out_filename = '-'
out_scale = 1
y_flip = 1

to_matlab = False

thickness_greater = 0
thickness_less = 1000


def parse_args():
    global path_filename, sample_rate_mm, \
        translation, out_filename, out_scale, y_flip, \
        thickness_greater, thickness_less, \
        to_matlab

    error = False
    for i, arg in enumerate(sys.argv):
        if '.svg' in arg:
            path_filename.append(arg)
        elif '-o' == arg:
            if i + 1 >= len(sys.argv):
                error = True
                break
            out_filename = sys.argv[i + 1]
        elif '-r' == arg:
            if i + 1 >= len(sys.argv):
                error = True
                break
            try:
                sample_rate_mm = float(sys.argv[i + 1])
            except ValueError:
                error = True
                break
        elif '-s' == arg:
            if i + 1 >= len(sys.argv):
                error = True
                break
            try:
                out_scale = float(sys.argv[i + 1])
            except ValueError:
                error = True
                break
        elif '-t' == arg:
            if i + 2 >= len(sys.argv):
                error = True
                break
            try:
                translation = (float(sys.argv[i + 1]), float(sys.argv[i + 2]))
            except ValueError:
                error = True
                break
        elif '-f' == arg:
            y_flip = -1
        elif '-g' == arg:
            if i + 1 >= len(sys.argv):
                error = True
                break
            try:
                thickness_greater = float(sys.argv[i + 1])
            except ValueError:
                error = True
                break
        elif '-m' == arg:
            to_matlab = True
        elif '-l' == arg:
            if i + 1 >= len(sys.argv):
                error = True
                break
            try:
                thickness_less = float(sys.argv[i + 1])
            except ValueError:
                error = True
                break

    if len(path_filename) == 0 or error:
        sys.exit(
            "Usage:\n\tsvg-to-scap.py [-r sample_rate_mm] [-s out_scale] [-t translation_x translation_y] [-g greater_thickness] [-l less_thickness] [-f] path_filename.svg -o out.svg|out.scap")


def resample_path(path, sample_rate):
    resampled_path = []

    path_len = [0]
    for p in path:
        path_len.append(path_len[-1] + p.length())

    sample_len = sample_rate * path_len[-1]
    # print(sample_len)

    acc_sample_len = 0
    c = 0
    local_poly = None
    while acc_sample_len < path_len[-1]:
        large_ind = [ind for (l, ind) in zip(
            path_len, range(0, len(path_len))) if l > acc_sample_len]

        # print(('\t%d: ' % c) + str(large_ind))

        if len(large_ind) < 1:
            break

        curr_ind = large_ind[0] - 1
        next_ind = large_ind[0]

        local_t = (acc_sample_len - path_len[curr_ind]) / \
            (path_len[next_ind] - path_len[curr_ind])

        if type(path[curr_ind]) is Arc:
            local_poly = path[curr_ind]
            resampled_path.append(
                (local_poly.point(local_t).real, local_poly.point(local_t).imag))
        else:
            local_poly = path[curr_ind].poly()
            resampled_path.append(
                (local_poly(local_t).real, local_poly(local_t).imag))

        acc_sample_len = acc_sample_len + sample_len
        c = c + 1

    if len(resampled_path) == 0:
        if type(path[-1]) is not Arc:
            resampled_path.append(
                (path[0].poly()(0).real, path[0].poly()(0).imag))
        else:
            resampled_path.append(
                (path[0].point(0).real, path[0].point(0).imag))

    if type(path[-1]) is not Arc:
        local_poly = path[-1].poly()
        resampled_path.append((local_poly(1).real, local_poly(1).imag))
    else:
        local_poly = path[-1]
        resampled_path.append(
            (local_poly.point(1).real, local_poly.point(1).imag))

    return resampled_path


def convert_path(path):
    converted_path = []
    for i in range(len(path)):
        if i == 0:
            converted_path.append((path[i].start.real, path[i].start.imag))
        converted_path.append((path[i].end.real, path[i].end.imag))

    return converted_path


def parse_path(paths, attributes, sample_rate_mm):
    resampled_paths = []
    attribute_maps = []

    for i in range(0, len(paths)):
        # It is a stroke not a clip path
        # if 'style' in attributes[i]:
        if len(paths[i]) == 0 or paths[i].length() == 0:
            continue

        if 'style' in attributes[i]:
            attr_str = attributes[i]['style'].split(';')
            attr_dict = {}
            for str in attr_str:
                temp_str = str.replace(' ', '').split(':')
                if len(temp_str) < 2:
                    continue
                attr_dict[temp_str[0]] = temp_str[1]
        else:
            attr_dict = {k: v for k,
                         v in attributes[i].items() if k != 'd' and k != 'id'}

        attribute_maps.append(attr_dict)

        seen_non_line = False
        for j in range(len(paths[i])):
            if type(paths[i][j]) is not Line:
                seen_non_line = True
                break

        if seen_non_line:
            path = resample_path(
                paths[i], sample_rate_mm / paths[i].length())
        else:
            path = convert_path(paths[i])

        if 'fill' in attr_dict:
            if path[0] != path[-1]:
                path.append(path[0])

        resampled_paths.append(path)

    return resampled_paths, attribute_maps


def build_bbox(resampled_paths):
    bbox = [(float('inf'), float('inf')), (-float('inf'), -float('inf'))]
    for i in range(len(resampled_paths)):
        path = resampled_paths[i]

        for point in path:
            bbox[0] = (min(bbox[0][0], point[0]), min(bbox[0][1], point[1]))
            bbox[1] = (max(bbox[1][0], point[0]), max(
                bbox[1][1], y_flip * point[1]))

    if bbox[0][0] == float('inf'):
        bbox = [(0, 0), (0, 0)]

    return bbox


def split_connected_components(path, attr):
    paths = []
    attrs = []
    curr_path = Path()
    for i in range(len(path)):
        if len(curr_path) > 0 and curr_path[-1].end != path[i].start:
            paths.append(curr_path)
            attrs.append({k: v for k, v in attr.items() if k != 'd'})
            curr_path = Path()
        curr_path.append(path[i])

    if len(curr_path) > 0:
        paths.append(curr_path)
        attrs.append({k: v for k, v in attr.items() if k != 'd'})

    return paths, attrs


def svg2poly(p_name):
    print('Converting: {}'.format(p_name))
    width, height = get_svg_size(p_name)

    if width == 0 or height == 0:
        return [], [], (0, 0)

    paths_in, attributes_in, _ = svg2paths2(p_name)
    # print(paths_in)

    paths = []
    attributes = []
    for i in range(len(paths_in)):
        path, attr = split_connected_components(paths_in[i], attributes_in[i])
        paths += path
        attributes += attr
    # print(len(paths))
    # print(len(attr))
    # exit()

    # filter 0-length path
    long_paths = []
    long_attributes = []
    for i, p in enumerate(paths):
        try:
            if p.length() > 0:
                long_paths.append(p)
                long_attributes.append(attributes[i])
        except ZeroDivisionError:
            print('Error path: ' + str(p))

    paths = long_paths
    attributes = long_attributes

    resampled_paths, attribute_maps = parse_path(
        paths, attributes, sample_rate_mm)

    # filter 0-length path
    long_paths = []
    long_attributes = []
    for i, p in enumerate(resampled_paths):
        try:
            if len(p) >= 2:
                long_paths.append(p)
                long_attributes.append(attribute_maps[i])
        except ZeroDivisionError:
            print('Error path: ' + str(p))

    return long_paths, long_attributes, (width, height)
