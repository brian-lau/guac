__author__ = 'dcard'

def make_header(title):
    header = '<!DOCTYPE HTML PUBLIC>\n'
    header += '<html>\n'
    header += '<head>\n'
    header += '<title>' + str(title) + '</title>\n'
    header += '<script src="../sorttable.js"></script>\n'
    header += '<link rel="stylesheet" type="text/css" href="../style.css">\n'
    header += '</head>\n'
    return header


def make_body_start():
    return '<body>\n'

def make_body_end():
    return '</body>\n'

def make_footer():
    return '</html>\n'

def make_heading(text, level=1):
    return '<h' + str(level) + '>' + text + '</h' + str(level) + '>'

def make_link(link, text):
    return '<a href="' + link + '">' + text + '</a>'

def make_table_start(col_widths=None, style='t1'):
    table = '<table class="' + style + '">\n'
    if col_widths is not None:
        for width in col_widths:
            table += '<col width="' + str(width) + '">\n'
    return table

def make_table_header(header):
    table = '<tr>'
    for h in header:
        table += '<th>' + h + '</th> '
    table += '</tr>\n'
    return table


def make_table_row(row, colours=None):
    table = '<tr>'
    for i, r in enumerate(row):
        if colours is not None:
            table += '<td style="color: rgb' + str(colours[i]) + '">' + r + '</td> '
        else:
            table += '<td>' + r + '</td> '
    table += '</tr>\n'
    return table

def make_table_end():
    return '</table>\n'