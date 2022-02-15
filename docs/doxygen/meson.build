doxygen = find_program('doxygen', required : false)

if doxygen.found()

  cdata = configuration_data()
  cdata.set('VERSION', meson.project_version())

  if find_program('dot', required : false).found()
    cdata.set('HAVE_DOT', 'YES')
  else
    cdata.set('HAVE_DOT', 'NO')
  endif

  cdata.set('PROJECT_NAME',meson.project_name())
  cdata.set('PROJECT_LOGO', join_paths(meson.current_source_dir()))
  cdata.set('IMAGE_PATH', join_paths(meson.current_source_dir(), 'images'))
  cdata.set('enable_html_docs', true)
  cdata.set('enable_latex_docs', false)
  cdata.set('enable_xml_docs', true)
  cdata.set('top_builddir', join_paths(meson.build_root()))
  cdata.set('top_srcdir', join_paths(meson.source_root()))
  cdata.set('abs_top_srcdir', join_paths(meson.source_root()))
  cdata.set('CURRENT_SRC_DIR', join_paths(meson.current_source_dir()))

  doxyfile = configure_file(input: 'Doxyfile.in',
                            output: 'Doxyfile',
                            configuration: cdata,
                            install: false)

  datadir = join_paths(get_option('datadir'), 'doc','gnuradio')

  # TODO: Figure out how to install docs
  html_target = custom_target('doc',
			      build_always_stale : not meson.is_subproject(),
                              input: doxyfile,
                              output: 'html',
                              command: [doxygen, doxyfile],
                              install: false,
                              install_dir: datadir)

endif
