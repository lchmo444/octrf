import Options

subdirs = 'src'

def options(opt):
    opt.tool_options('compiler_cxx')

def configure(conf):
    conf.check_tool('compiler_cxx')
    conf.env.CXXFLAGS += ['-W', '-Wall', '-Wextra', '-O2', '-g', '-Wno-sign-compare']
    conf.recurse(subdirs)
    #conf.check_cfg(package = 'opencv', args='--cflags --libs', atleast_version='2.2.0', uselib_store='opencv')

def build(bld):
    bld.recurse(subdirs)
    bld(features = 'cxx cprogram',
        source = 'sandbox.cpp',
        target = 'sandbox',
        uselib_local = 'octrf',
        includes = 'src')

