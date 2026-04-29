SECTIONS {
 /DISCARD/ : {
  *(.discard)
  *(.discard.*)
 }
 __ksymtab 0 : { *(SORT(___ksymtab+*)) }
 __ksymtab_gpl 0 : { *(SORT(___ksymtab_gpl+*)) }
 __kcrctab 0 : { *(SORT(___kcrctab+*)) }
 __kcrctab_gpl 0 : { *(SORT(___kcrctab_gpl+*)) }
 .ctors 0 : ALIGN(8) { *(SORT(.ctors.*)) *(.ctors) }
 .init_array 0 : ALIGN(8) { *(SORT(.init_array.*)) *(.init_array) }
 .altinstructions 0 : ALIGN(8) { KEEP(*(.altinstructions)) }
 __bug_table 0 : ALIGN(8) { KEEP(*(__bug_table)) }
 __jump_table 0 : ALIGN(8) { KEEP(*(__jump_table)) }
 __patchable_function_entries : { *(__patchable_function_entries) }
}
SECTIONS {
 .plt : { BYTE(0) }
 .got : { BYTE(0) }
 .got.plt : { BYTE(0) }
}
